from torch import nn
import torch
from loss import ContrastiveWeight, AggregationRebuild, AutomaticWeightedLoss, SignalDiceLoss, mae_loss, dtw_loss
from metrics import SignalDice as SDSC


class TFC(nn.Module):
    def __init__(self, configs, args):
        super(TFC, self).__init__()
        self.training_mode = args.training_mode

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.dense = nn.Sequential(
            nn.Linear(configs.CNNoutput_channel * configs.final_out_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        if self.training_mode == 'pre_train':
            # self.awl = AutomaticWeightedLoss(2)
            self.contrastive = ContrastiveWeight(args)
            self.aggregation = AggregationRebuild(args)
            self.head = nn.Linear(1280, 178)
            # self.mse = torch.nn.MSELoss()
        
            # SET LOSS MODE
            self.loss_mode = args.loss_mode
            self.mse  = torch.nn.MSELoss()
            self.sdsc = SignalDiceLoss()
            self.mae  = mae_loss()            
            self.dtw = dtw_loss(approx=True)

            if self.loss_mode == "hybrid":
                self.awl = AutomaticWeightedLoss(3)
            else:
                self.awl = AutomaticWeightedLoss(2)

            self.sdsc_metric = SDSC()

    def forward(self, x_in_t, pretrain=False):
        if pretrain:
            x = self.conv_block1(x_in_t)
            x = self.conv_block2(x)
            x = self.conv_block3(x)

            h = x.reshape(x.shape[0], -1)
            z = self.dense(h)

            loss_cl, similarity_matrix, logits, positives_mask = self.contrastive(z)
            rebuild_weight_matrix, agg_x = self.aggregation(similarity_matrix, x)
            pred_x = self.head(agg_x.reshape(agg_x.size(0), -1))

            # series reconstruction
            loss_rb = self.mse(pred_x, x_in_t.reshape(x_in_t.size(0), -1).detach())
            loss_sd = self.sdsc(pred_x, x_in_t.reshape(x_in_t.size(0), -1).detach())
            loss_mae = self.mae(pred_x, x_in_t.reshape(x_in_t.size(0), -1).detach())
            loss_dtw = self.dtw(pred_x.unsqueeze(1), x_in_t.reshape(x_in_t.size(0), 1, -1).detach())


            if self.loss_mode == "mse":
                loss = self.awl(loss_cl, loss_rb)
            elif self.loss_mode =="sdsc":
                loss = self.awl(loss_cl, loss_sd)
            elif self.loss_mode =="mae":
                loss = self.awl(loss_cl, loss_mae)
            elif self.loss_mode == "dtw":
                loss = self.awl(loss_dtw)
            else:
                loss = self.awl(loss_cl, loss_rb, loss_sd)

            # metrics 
            metric_sd = self.sdsc_metric(pred_x, x_in_t.reshape(x_in_t.size(0), -1).detach())

            # loss_rb = self.mse(pred_x, x_in_t.reshape(x_in_t.size(0), -1).detach())
            # loss = self.awl(loss_cl, loss_rb)

            return loss, loss_cl, loss_rb, loss_sd, loss_mae, loss_dtw, metric_sd
        else:
            x = self.conv_block1(x_in_t)
            x = self.conv_block2(x)
            x = self.conv_block3(x)

            h = x.reshape(x.shape[0], -1)
            z = self.dense(h)

            return h, z


class target_classifier(nn.Module):  # Classification head
    def __init__(self, configs):
        super(target_classifier, self).__init__()
        self.logits = nn.Linear(1280, 64)
        self.logits_simple = nn.Linear(64, configs.num_classes_target)

    def forward(self, emb):
        """2-layer MLP"""
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)
        return pred
