from torch import nn
import torch
from loss import ContrastiveWeight, AggregationRebuild, AutomaticWeightedLoss, SignalDiceLoss, mae_loss, dtw_loss, DTWLoss, DiffZCRLoss
from metrics import SignalDice as SDSC, pearson_correlation, si_snr


class TFC(nn.Module):
    def __init__(self, configs, args):
        super(TFC, self).__init__()
        self.training_mode = args.training_mode
        self.configs = configs

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

        if self.training_mode == 'pre_train' or self.training_mode == 'bench_mark':
            # self.awl = AutomaticWeightedLoss(2)
            self.contrastive = ContrastiveWeight(args)
            self.aggregation = AggregationRebuild(args)
            self.head = nn.Linear(1280, 178)
            # self.mse = torch.nn.MSELoss()
        
            # SET LOSS MODE
            self.loss_mode = args.loss_mode
            self.mse  = torch.nn.MSELoss()
            self.sdsc = SignalDiceLoss(alpha=args.alpha)
            self.mae  = mae_loss()
            self.dtw = dtw_loss(approx=True,use_cuda=False)
            self.zcr = DiffZCRLoss(alpha=10.0)  # AAAI27 Plan B++ AC-CL-2

            if self.loss_mode == "hybrid":
                self.awl = AutomaticWeightedLoss(3)
            else:
                self.awl = AutomaticWeightedLoss(2)

            self.sdsc_metric = SDSC()
            self.pcc = pearson_correlation
            self.si_snr = si_snr

    def one_step(self, x_in_t):
        x = self.conv_block1(x_in_t)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        h = x.reshape(x.shape[0], -1)
        z = self.dense(h)


        loss_cl, similarity_matrix, logits, positives_mask = self.contrastive(z)
        rebuild_weight_matrix, agg_x = self.aggregation(similarity_matrix, x)
        pred_x = self.head(agg_x.reshape(agg_x.size(0), -1))
        b, h, s = x_in_t.shape
        return pred_x, loss_cl


    def bench_mark(self):
        # 1. 테스트할 시퀀스 길이 목록
        seq_lengths = [256, 512, 1024, 2048, 4096]
        batch_size = 16
        
        # 모델의 설정 가져오기 (없으면 기본값)
        input_ch = getattr(self, 'configs', None).input_channels if hasattr(self, 'configs') else 1
        final_ch = getattr(self, 'configs', None).final_out_channels if hasattr(self, 'configs') else 64
        
        # self.head의 입력 차원 (1280)을 가져오기 위해 현재 레이어 확인
        head_in_features = self.head.in_features if hasattr(self.head, 'in_features') else 1280

        times_ours = []
        mems_ours = []
        
        # 현재 디바이스
        device = next(self.parameters()).device 

        # Loss 함수 매핑
        if self.loss_mode == "mse":
            loss_fn = self.mse
        elif self.loss_mode =="sdsc":
            loss_fn = self.sdsc
        elif self.loss_mode =="mae":
            loss_fn = self.mae
        elif self.loss_mode == "dtw":
            loss_fn = self.dtw
        elif self.loss_mode == 'pcc':
            loss_fn = self.pcc
        elif self.loss_mode == 'snr':
            loss_fn = self.si_snr
        else:
            # 기본값
            loss_fn = self.mse

        print(f"=== Benchmarking Mode: {self.loss_mode} ===")
        print(f"{'Length':<10} | {'Time (ms)':<15} | {'Memory (MB)':<15}")
        print("-" * 45)

        # 원래 레이어 백업 (테스트 끝나고 복구용)
        original_dense = self.dense
        original_head = self.head

        for L in seq_lengths:
            try:
                # ---------------------------------------------------------
                # [중요] 현재 길이 L에 맞춰 모델 레이어 동적 수정 (Patching)
                # ---------------------------------------------------------

                self.conv_block1 = nn.Sequential(
                        nn.Conv1d(L, 32, kernel_size=self.configs.kernel_size,
                                stride=self.configs.stride, bias=False, padding=(self.configs.kernel_size // 2)),
                        nn.BatchNorm1d(32),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
                        nn.Dropout(self.configs.dropout)
                    ).to(device)
                 
                # 1. Conv 블록을 3번 통과하면 길이는 대략 L / 8이 됨 (MaxPool 3번)
                #    정확한 계산: L -> L/2 -> L/4 -> L/8
                # feat_len = L // 8
                
                # 2. Dense 레이어 입력 크기 계산 (Flatten size)
                # flat_size = feat_len * final_ch
                
                # 3. Dense 레이어 교체 (입력 크기 맞춤)
                #    기존 구조: Linear -> BN -> ReLU -> Linear
                #    여기서는 벤치마크용이므로 단순 Linear로 교체하거나 구조 유지
                self.dense = nn.Sequential(
                    nn.Linear(256, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Linear(256, 128) # 출력 128은 유지 (z의 차원)
                ).to(device)

                # 4. Head 레이어 교체 (출력 크기를 L로 맞춤)
                #    입력 차원(1280)은 aggregation 로직에 따라 다를 수 있으나
                #    __init__에 1280으로 하드코딩 되어 있었으므로 유지한다고 가정
                self.head = nn.Linear(256, L).to(device)
                # ---------------------------------------------------------

                # 입력 데이터 생성
                dummy_input = torch.randn(batch_size, L, input_ch ).to(device)
                
                # 메모리 초기화
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize()

                # Warm-up
                self.train()
                for _ in range(5):
                    self.zero_grad()
                    pred_x, loss_cl = self.one_step(dummy_input)
                    if self.loss_mode == 'dtw':
                         loss = loss_fn(pred_x.reshape(batch_size, L,-1), dummy_input)
                    else:
                        loss = loss_fn(pred_x, dummy_input.reshape(dummy_input.size(0), -1).detach())
                    loss.backward()

                # 실제 측정
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                for _ in range(50): 
                    self.zero_grad() # 필수
                    
                    pred_x, loss_cl = self.one_step(dummy_input)
                    
                    if self.loss_mode == 'dtw':
                        target_loss = loss_fn(pred_x.reshape(batch_size,  L,-1), dummy_input)
                    else:
                        target_loss = loss_fn(pred_x, dummy_input.reshape(dummy_input.size(0), -1).detach())
                    
                    # AWL 적용
                    if self.loss_mode in ['pcc', 'snr']:
                         loss = self.awl(loss_cl, -target_loss)
                    else:
                         loss = self.awl(loss_cl, target_loss)
                         
                    loss.backward() 

                end_event.record()
                torch.cuda.synchronize()
                
                avg_time = start_event.elapsed_time(end_event) / 50 
                max_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024 
                
                times_ours.append(avg_time)
                mems_ours.append(max_mem)
                
                print(f"{L:<10} | {avg_time:<15.2f} | {max_mem:<15.2f}")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"{L:<10} | {'OOM':<15} | {'OOM':<15}")
                    torch.cuda.empty_cache()
                    break 
                else:
                    # Shape mismatch 등 다른 에러 확인용
                    print(f"Error at L={L}: {e}")
                    break
        
        # 테스트 종료 후 원래 레이어로 복구 (선택 사항)
        self.dense = original_dense
        self.head = original_head
        print("\n")

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
            b, h, s = x_in_t.shape

            # series reconstruction
            loss_rb = self.mse(pred_x, x_in_t.reshape(x_in_t.size(0), -1).detach())
            loss_sd = self.sdsc(pred_x, x_in_t.reshape(x_in_t.size(0), -1).detach())
            loss_mae = self.mae(pred_x, x_in_t.reshape(x_in_t.size(0), -1).detach())
            loss_dtw = self.dtw(pred_x.reshape(x_in_t.size(0), h, s), x_in_t.detach())
            # loss_dtw = torch.tensor([0.,]).to(0)
            


            # metrics 
            metric_sd     = self.sdsc_metric(pred_x, x_in_t.reshape(x_in_t.size(0), -1).detach())
            metric_pcc    = self.pcc(pred_x, x_in_t.reshape(x_in_t.size(0), -1).detach())
            metric_si_snr = self.si_snr(pred_x, x_in_t.reshape(x_in_t.size(0), -1).detach())

            if self.loss_mode == "mse":
                loss = self.awl(loss_cl, loss_rb)
            elif self.loss_mode =="sdsc":
                loss = self.awl(loss_cl, loss_sd)
            elif self.loss_mode =="mae":
                loss = self.awl(loss_cl, loss_mae)
            elif self.loss_mode == "dtw":
                loss = self.awl(loss_dtw)
            elif self.loss_mode == 'pcc':
                loss = self.awl(loss_cl, (1-metric_pcc))
            elif self.loss_mode == 'snr':
                loss = self.awl(loss_cl, -metric_si_snr)
            elif self.loss_mode == 'zcr':
                # AAAI27 Plan B++ AC-CL-2 — DiffZCRLoss baseline
                loss_zcr = self.zcr(pred_x, x_in_t.reshape(x_in_t.size(0), -1).detach())
                loss = self.awl(loss_cl, loss_zcr)
            else:
                loss = self.awl(loss_cl, loss_rb, loss_sd)
                # loss = (loss_cl + loss_rb*0.5 + loss_sd*0.5).mean()

            # loss_rb = self.mse(pred_x, x_in_t.reshape(x_in_t.size(0), -1).detach())
            # loss = self.awl(loss_cl, loss_rb)

            return loss, loss_cl, loss_rb, loss_sd, loss_mae, loss_dtw, metric_sd, metric_pcc, metric_si_snr
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
