from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, transfer_weights, show_series, show_matrix
from utils.augmentations import masked_data
from utils.metrics import metric, SignalDice as _SignalDiceMetric
from utils.losses import dtw_loss as _DTWLossEval, DILATELoss as _DILATELossEval
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from collections import OrderedDict
from tensorboardX import SummaryWriter
import random

warnings.filterwarnings('ignore')

class Exp_SimMTM(Exp_Basic):
    def __init__(self, args):
        super(Exp_SimMTM, self).__init__(args)
        # US-005 / A1-NEW-2: include loss_mode, alpha tag, and seed in TensorBoard path
        # so 25 runs from the 5-loss x 5-seed sweep do not collide in the same directory.
        alpha_tag = str(args.alpha) if getattr(args, 'alpha', None) is not None else "default"
        seed_tag = getattr(args, 'seed', 'unspecified')
        self.writer = SummaryWriter(
            f"./outputs/logs/{args.data}/{args.loss_mode}/alpha-{alpha_tag}/seed-{seed_tag}"
        )
        self.patience = 5
        # AMP scaler. Enabled only when --use_amp is set; otherwise no-op (FP32 path).
        # Crucial for ECL/Traffic memory: attention scores in FP16 are half-size, which
        # turned a 48GB OOM on Traffic batch=2 into comfortable headroom.
        self.use_amp = bool(getattr(args, 'use_amp', False)) and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)



    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        print(torch.cuda.device_count())
        if self.args.load_checkpoints:
            print("Loading ckpt: {}".format(self.args.load_checkpoints))

            model = transfer_weights(self.args.load_checkpoints, model, device=self.device)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!", self.args.device_ids)
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        # print out the model size
        print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def pretrain(self):

        # data preparation
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        earlystopping_counts = 0

        # show cases
        self.train_show = next(iter(train_loader))
        self.valid_show = next(iter(vali_loader))
        # US-005 Task 5b / C2: when alpha is None (default), nest checkpoint by loss_mode
        # so the finetune load path os.path.join(..., args.data, args.loss_mode, ...) at
        # run.py:170 actually finds the saved ckpt_best.pth.
        if self.args.alpha is not None:
            path = os.path.join(self.args.pretrain_checkpoints, self.args.data, str(self.args.alpha))
        else:
            path = os.path.join(self.args.pretrain_checkpoints, self.args.data, self.args.loss_mode)
        if not os.path.exists(path):
            os.makedirs(path)

        # optimizer
        model_optim = self._select_optimizer()
        #model_optim.add_param_group({'params': self.awl.parameters(), 'weight_decay': 0})
        model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optim,
                                                                     T_max=self.args.train_epochs)

        # pre-training
        min_vali_loss = None
        for epoch in range(self.args.train_epochs):
            start_time = time.time()

            (train_loss, train_cl_loss, train_rb_loss, train_sdsc_loss,
             train_mae_loss, train_dtw_loss, train_zcr_loss, train_dilate_loss,
             train_sdsc_metric, train_pcc_metric, train_si_snr_metric) = self.pretrain_one_epoch(train_loader, model_optim, model_scheduler)
            (valid_loss, valid_cl_loss, valid_rb_loss, valid_sdsc_loss,
             valid_mae_loss, valid_dtw_loss, valid_zcr_loss, valid_dilate_loss,
             valid_sdsc_metric, valid_pcc_metric, valid_si_snr_metric) = self.valid_one_epoch(vali_loader, corr=True)

            # log and Loss (zcr/dilate appended after dtw)
            end_time = time.time()
            print(
                "Epoch: {epoch}, Lr: {lr:.7f}, Time: {dt:.2f}s | "
                "Train cl/rb/sd/mae/dtw/zcr/dilate: "
                "{tcl:.4f}/{trb:.4f}/{tsd:.4f}/{tmae:.4f}/{tdtw:.4f}/{tzcr:.4f}/{tdil:.4f} | "
                "Val cl/rb/sd/mae/dtw/zcr/dilate: "
                "{vcl:.4f}/{vrb:.4f}/{vsd:.4f}/{vmae:.4f}/{vdtw:.4f}/{vzcr:.4f}/{vdil:.4f} | "
                "Metric tr SDSC/PCC/SI-SNR: {tsdm:.4f}/{tpccm:.4f}/{tsnrm:.4f} "
                "val SDSC/PCC/SI-SNR: {vsdm:.4f}/{vpccm:.4f}/{vsnrm:.4f}".format(
                    epoch=epoch, lr=model_scheduler.get_lr()[0], dt=end_time - start_time,
                    tcl=train_cl_loss, trb=train_rb_loss, tsd=train_sdsc_loss,
                    tmae=train_mae_loss, tdtw=train_dtw_loss, tzcr=train_zcr_loss, tdil=train_dilate_loss,
                    vcl=valid_cl_loss, vrb=valid_rb_loss, vsd=valid_sdsc_loss,
                    vmae=valid_mae_loss, vdtw=valid_dtw_loss, vzcr=valid_zcr_loss, vdil=valid_dilate_loss,
                    tsdm=train_sdsc_metric, tpccm=train_pcc_metric, tsnrm=train_si_snr_metric,
                    vsdm=valid_sdsc_metric, vpccm=valid_pcc_metric, vsnrm=valid_si_snr_metric,
                )
            )

            loss_scalar_dict = {
                'train_loss': train_loss,
                'train_cl_loss': train_cl_loss,
                'train_rb_loss': train_rb_loss,
                'train_sdsc_loss': train_sdsc_loss,
                'train_mae_loss': train_mae_loss,
                'train_dtw_loss': train_dtw_loss,
                'train_zcr_loss': train_zcr_loss,
                'train_dilate_loss': train_dilate_loss,
                'train_sdsc': train_sdsc_metric,
                'train_pcc_metric': train_pcc_metric,
                'train_si_snr_metric': train_si_snr_metric,
                'vali_loss': valid_loss,
                'valid_cl_loss': valid_cl_loss,
                'valid_rb_loss': valid_rb_loss,
                'valid_sdsc_loss': valid_sdsc_loss,
                'valid_mae_loss': valid_mae_loss,
                'valid_dtw_loss': valid_dtw_loss,
                'valid_zcr_loss': valid_zcr_loss,
                'valid_dilate_loss': valid_dilate_loss,
                'valid_sdsc': valid_sdsc_metric,
                'valid_pcc_metric': valid_pcc_metric,
                'valid_si_snr_metric': valid_si_snr_metric,
            }

            self.writer.add_scalars(f"pretrain_loss", loss_scalar_dict, epoch)

            # checkpoint saving
            if not min_vali_loss or valid_loss <= min_vali_loss:
                if epoch == 0:
                    min_vali_loss = valid_loss

                print(
                    "Validation loss decreased ({0:.4f} --> {1:.4f}).  Saving model epoch{2} ...".format(min_vali_loss, valid_loss, epoch))

                min_vali_loss = valid_loss
                self.encoder_state_dict = OrderedDict()
                for k, v in self.model.state_dict().items():
                    if 'encoder' in k or 'enc_embedding' in k:
                        if 'module.' in k:
                            k = k.replace('module.', '')  # multi-gpu
                        self.encoder_state_dict[k] = v
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.encoder_state_dict}
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt_best.pth"))
                earlystopping_counts = 0
            else:
                earlystopping_counts += 1

            if (epoch + 1) % 10 == 0:
                print("Saving model at epoch {}...".format(epoch + 1))

                self.encoder_state_dict = OrderedDict()
                for k, v in self.model.state_dict().items():
                    if 'encoder' in k or 'enc_embedding' in k:
                        if 'module.' in k:
                            k = k.replace('module.', '')
                        self.encoder_state_dict[k] = v
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.encoder_state_dict}
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt{epoch + 1}.pth"))

                self.show(5, epoch + 1, 'train')
                self.show(5, epoch + 1, 'valid')

            if earlystopping_counts == self.patience:
                break
    



    def pretrain_one_epoch(self, train_loader, model_optim, model_scheduler):

        train_loss = []
        train_cl_loss = []
        train_rb_loss = []
        train_sdsc_loss = []
        train_mae_loss  = []
        train_dtw_loss  = []
        train_zcr_loss = []
        train_dilate_loss = []
        train_sdsc_metric =[]
        train_pcc_metric = []
        train_si_snr_metric = []

        self.model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            model_optim.zero_grad()

            if self.args.select_channels < 1:

                # random select channels
                B, S, C = batch_x.shape
                random_c = int(C * self.args.select_channels)
                if random_c < 1:
                    random_c = 1

                index = torch.LongTensor(random.sample(range(C), random_c))
                batch_x = torch.index_select(batch_x, 2, index)

            # data augumentation
            batch_x_m, batch_x_mark_m, mask = masked_data(batch_x, batch_x_mark, self.args.mask_rate, self.args.lm, self.args.positive_nums)
            batch_x_om = torch.cat([batch_x, batch_x_m], 0)

            batch_x = batch_x.float().to(self.device)
            # batch_x_mark = batch_x_mark.float().to(self.device)

            # masking matrix
            mask = mask.to(self.device)
            mask_o = torch.ones(size=batch_x.shape).to(self.device)
            mask_om = torch.cat([mask_o, mask], 0).to(self.device)

            # to device
            batch_x = batch_x.float().to(self.device)
            batch_x_om = batch_x_om.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)

            # encoder (15 values: ..., loss_zcr, loss_dilate, metrics, masks/logits/...)
            # AMP autocast wraps the model forward + loss compute.
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                (loss, loss_cl, loss_rb, loss_sd, loss_mae, loss_dtw,
                 loss_zcr, loss_dilate,
                 metric_sd, metric_pcc, metric_si_snr,
                 _, _, _, _) = self.model(batch_x_om, batch_x_mark, batch_x, mask=mask_om)

                # Gathering (still inside autocast; cheap)
                loss      = loss.mean()
                loss_cl   = loss_cl.mean()
                loss_rb   = loss_rb.mean()
                loss_sd   = loss_sd.mean()
                loss_mae  = loss_mae.mean()
                loss_dtw  = loss_dtw.mean()
                loss_zcr  = loss_zcr.mean()
                loss_dilate = loss_dilate.mean()
                metric_sd = metric_sd.mean()
                metric_pcc = metric_pcc.mean()
                metric_si_snr = metric_si_snr.mean()

            # backward (GradScaler is a no-op when use_amp=False)
            self.scaler.scale(loss).backward()
            self.scaler.step(model_optim)
            self.scaler.update()

            # record
            train_loss.append(loss.item())
            train_cl_loss.append(loss_cl.item())
            train_rb_loss.append(loss_rb.item())
            train_sdsc_loss.append(loss_sd.item())
            train_mae_loss.append(loss_mae.item())
            train_dtw_loss.append(loss_dtw.item())
            train_zcr_loss.append(loss_zcr.item())
            train_dilate_loss.append(loss_dilate.item())
            train_sdsc_metric.append(metric_sd.item())
            train_pcc_metric.append(metric_pcc.item())
            train_si_snr_metric.append(metric_si_snr.item())

        model_scheduler.step()

        train_loss        = np.average(train_loss)
        train_cl_loss     = np.average(train_cl_loss)
        train_rb_loss     = np.average(train_rb_loss)
        train_sdsc_loss   = np.average(train_sdsc_loss)
        train_mae_loss    = np.average(train_mae_loss)
        train_dtw_loss    = np.average(train_dtw_loss)
        train_zcr_loss    = np.average(train_zcr_loss)
        train_dilate_loss = np.average(train_dilate_loss)
        train_sdsc_metric = np.average(train_sdsc_metric)
        train_pcc_metric  = np.average(train_pcc_metric)
        train_si_snr_metric = np.average(train_si_snr_metric)

        return (train_loss, train_cl_loss, train_rb_loss, train_sdsc_loss,
                train_mae_loss, train_dtw_loss, train_zcr_loss, train_dilate_loss,
                train_sdsc_metric, train_pcc_metric, train_si_snr_metric)

    def valid_one_epoch(self, vali_loader, corr=False):
        valid_loss = []
        valid_cl_loss = []
        valid_rb_loss = []
        valid_sdsc_loss = []
        valid_mae_loss  = []
        valid_dtw_loss  = []
        valid_zcr_loss = []
        valid_dilate_loss = []
        valid_sdsc_metric = []
        valid_pcc_metric = []
        valid_si_snr_metric = []

        self.model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            # data augumentation
            batch_x_m, batch_x_mark_m, mask = masked_data(batch_x, batch_x_mark, self.args.mask_rate, self.args.lm,
                                                          self.args.positive_nums)
            batch_x_om = torch.cat([batch_x, batch_x_m], 0)

            # masking matrix
            mask = mask.to(self.device)
            mask_o = torch.ones(size=batch_x.shape).to(self.device)
            mask_om = torch.cat([mask_o, mask], 0).to(self.device)

            # to device
            batch_x = batch_x.float().to(self.device)
            batch_x_om = batch_x_om.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)

            # encoder (15 values)
            (loss, loss_cl, loss_rb, loss_sd, loss_mae, loss_dtw,
             loss_zcr, loss_dilate,
             metric_sd, metric_pcc, metric_si_snr,
             _, _, _, _) = self.model(batch_x_om, batch_x_mark, batch_x, mask=mask_om)

            # Gathering
            loss      = loss.mean()
            loss_cl   = loss_cl.mean()
            loss_rb   = loss_rb.mean()
            loss_sd   = loss_sd.mean()
            loss_mae  = loss_mae.mean()
            loss_dtw  = loss_dtw.mean()
            loss_zcr  = loss_zcr.mean()
            loss_dilate = loss_dilate.mean()
            metric_sd = metric_sd.mean()
            metric_pcc = metric_pcc.mean()
            metric_si_snr = metric_si_snr.mean()


            # Record
            valid_loss.append(loss.item())
            valid_cl_loss.append(loss_cl.item())
            valid_rb_loss.append(loss_rb.item())
            valid_sdsc_loss.append(loss_sd.item())
            valid_mae_loss.append(loss_mae.item())
            valid_dtw_loss.append(loss_dtw.item())
            valid_zcr_loss.append(loss_zcr.item())
            valid_dilate_loss.append(loss_dilate.item())
            valid_sdsc_metric.append(metric_sd.item())
            valid_pcc_metric.append(metric_pcc.item())
            valid_si_snr_metric.append(metric_si_snr.item())


        if corr is True:
            with open("./outputs/sample_result.txt", "w") as f:
                for mse, sdsc in zip(valid_rb_loss, valid_sdsc_metric):
                    f.write(f"{mse} {sdsc} {valid_pcc_metric} {valid_si_snr_metric} \n")
        vali_loss         = np.average(valid_loss)
        valid_cl_loss     = np.average(valid_cl_loss)
        valid_rb_loss     = np.average(valid_rb_loss)
        valid_sdsc_loss   = np.average(valid_sdsc_loss)
        valid_mae_loss    = np.average(valid_mae_loss)
        valid_dtw_loss    = np.average(valid_dtw_loss)
        valid_zcr_loss    = np.average(valid_zcr_loss)
        valid_dilate_loss = np.average(valid_dilate_loss)
        valid_sdsc_metric = np.average(valid_sdsc_metric)
        valid_pcc_metric  = np.average(valid_pcc_metric)
        valid_si_snr_metric = np.average(valid_si_snr_metric)
        self.model.train()
        return (vali_loss, valid_cl_loss, valid_rb_loss, valid_sdsc_loss,
                valid_mae_loss, valid_dtw_loss, valid_zcr_loss, valid_dilate_loss,
                valid_sdsc_metric, valid_pcc_metric, valid_si_snr_metric)

    def train(self, setting):

        # data preparation
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # Optimizer
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            start_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                if self.args.select_channels < 1:
                    # Random select channels
                    B, S, C = batch_x.shape
                    random_c = int(C * self.args.select_channels)
                    if random_c < 1:
                        random_c = 1

                    index = torch.LongTensor(random.sample(range(C), random_c))
                    batch_x = torch.index_select(batch_x, 2, index)
                    batch_y = torch.index_select(batch_y, 2, index)

                # to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # encoder (under AMP autocast so finetune also runs FP16 when --use_amp)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(batch_x, batch_x_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0

                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    loss = criterion(outputs, batch_y)

                self.scaler.scale(loss).backward()
                self.scaler.step(model_optim)
                self.scaler.update()

                # record
                train_loss.append(loss.item())

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            end_time = time.time()
            print(
            "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
                epoch + 1, train_steps, end_time - start_time, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        self.lr = model_optim.param_groups[0]['lr']

        return self.model

    def vali(self, vali_loader, criterion):
        total_loss = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # encoder
                outputs = self.model(batch_x, batch_x_mark)

                # loss
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)

                # record
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, loss_mode):
        test_data, test_loader = self._get_data(flag='test')

        preds = []
        trues = []
        folder_path = './outputs/test_results/{}'.format(self.args.data)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # encoder
                outputs = self.model(batch_x, batch_x_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        # Structural metrics on test forecasts.
        # SDSC: linear-time, evaluate once on the full tensor.
        sdsc_eval = _SignalDiceMetric()
        preds_t = torch.from_numpy(preds).float()
        trues_t = torch.from_numpy(trues).float()
        sdsc = float(sdsc_eval(preds_t, trues_t).item())

        # SoftDTW and DILATE: O(T^2) per sample. Run in chunks to keep memory bounded.
        # We compute on a fixed eval subset if test set is huge to keep wall-clock predictable.
        device = self.device if torch.cuda.is_available() else torch.device('cpu')
        EVAL_CHUNK = 64                                                # batches of 64 samples
        MAX_EVAL_SAMPLES = min(preds.shape[0], 1024)                  # cap at 1024 to keep test < 1 min
        eval_preds = preds_t[:MAX_EVAL_SAMPLES].to(device)
        eval_trues = trues_t[:MAX_EVAL_SAMPLES].to(device)

        softdtw_fn = _DTWLossEval(approx=True, use_cuda=torch.cuda.is_available())
        dilate_fn = _DILATELossEval(gamma_dilate=0.5, gamma_sdtw=0.01,
                                    use_cuda=torch.cuda.is_available())

        sdtw_vals, dilate_vals = [], []
        with torch.no_grad():
            for start in range(0, MAX_EVAL_SAMPLES, EVAL_CHUNK):
                end = min(start + EVAL_CHUNK, MAX_EVAL_SAMPLES)
                p_chunk = eval_preds[start:end]
                t_chunk = eval_trues[start:end]
                # dtw_loss.forward defaults to divergence form; pass div=False to get raw SoftDTW.
                sdtw_v = softdtw_fn(p_chunk, t_chunk, div=False).item()
                dlt_v  = dilate_fn(p_chunk, t_chunk).item()
                sdtw_vals.append((end - start) * sdtw_v)
                dilate_vals.append((end - start) * dlt_v)
        softdtw_v = sum(sdtw_vals) / MAX_EVAL_SAMPLES
        dilate_v  = sum(dilate_vals) / MAX_EVAL_SAMPLES

        print('{0}->{1}, mse:{2:.4f}, mae:{3:.4f}, sdsc:{4:.4f}, softdtw:{5:.4f}, dilate:{6:.4f}'.format(
            self.args.seq_len, self.args.pred_len, mse, mae, sdsc, softdtw_v, dilate_v))
        # 6-col format: "{seq}->{pred}, mse, mae, sdsc, softdtw, dilate".
        # analyze_multi_v2.py tolerates 3/4/6-col lines.
        out_line = '{0}->{1}, {2:.4f}, {3:.4f}, {4:.4f}, {5:.4f}, {6:.4f}\n'.format(
            self.args.seq_len, self.args.pred_len, mse, mae, sdsc, softdtw_v, dilate_v)
        with open(f"./outputs/{loss_mode}_score.txt", 'a') as f:
            f.write(out_line)
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, f"{loss_mode}_score.txt"), 'a') as f:
            f.write(out_line)

    def show(self, num, epoch, type='valid'):

        # show cases
        if type == 'valid':
            batch_x, batch_y, batch_x_mark, batch_y_mark = self.valid_show
        else:
            batch_x, batch_y, batch_x_mark, batch_y_mark = self.train_show

        # data augumentation
        batch_x_m, batch_x_mark_m, mask = masked_data(batch_x, batch_x_mark, self.args.mask_rate, self.args.lm,
                                                      self.args.positive_nums)
        batch_x_om = torch.cat([batch_x, batch_x_m], 0)

        # masking matrix
        mask = mask.to(self.device)
        mask_o = torch.ones(size=batch_x.shape).to(self.device)
        mask_om = torch.cat([mask_o, mask], 0).to(self.device)

        # to device
        batch_x = batch_x.float().to(self.device)
        batch_x_om = batch_x_om.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)

        # Encoder (15 values incl. loss_zcr, loss_dilate)
        with torch.no_grad():
            (loss, loss_cl, loss_rb, loss_sd, loss_mae, loss_dtw,
             loss_zcr, loss_dilate,
             metric_sd, metric_pcc, metric_si_snr,
             positives_mask, logits, rebuild_weight_matrix, pred_batch_x) = self.model(batch_x_om, batch_x_mark, batch_x, mask=mask_om)

        for i in range(num):

            if i >= batch_x.shape[0]:
                continue

        fig_logits, fig_positive_matrix, fig_rebuild_weight_matrix = show_matrix(logits, positives_mask, rebuild_weight_matrix)
        self.writer.add_figure(f"/{type} show logits_matrix", fig_logits, global_step=epoch)
        self.writer.add_figure(f"/{type} show positive_matrix", fig_positive_matrix, global_step=epoch)
        self.writer.add_figure(f"/{type} show rebuild_weight_matrix", fig_rebuild_weight_matrix, global_step=epoch)