"""Per-dataset hyperparameter table for the multi-dataset SDSC sweep.

Extracted from the existing scripts/pretrain/{dataset}.sh and
scripts/finetune/{dataset}.sh files so each run reproduces the published config.
A bash wrapper imports this Python module to get the right --flags for run.py.

Source files inspected (2026-05-21):
    scripts/pretrain/ETT_script/{ETTh1,ETTh2,ETTm1,ETTm2}.sh
    scripts/pretrain/Weather_script/Weather.sh
    scripts/pretrain/ECL_script/ECL.sh
    scripts/pretrain/Traffic_script/Traffic.sh
    scripts/finetune/ETT_script/{ETTh1,ETTh2,ETTm1,ETTm2}.sh
    scripts/finetune/Weather_script/Weather.sh
    scripts/finetune/ECL_script/ECL.sh
    scripts/finetune/Traffic/Traffic.sh

For each dataset we keep:
    data_root, data_path, data, enc_in/dec_in/c_out (= channel count),
    pretrain hyperparams (e_layers, n_heads, d_model, d_ff, positive_nums,
    mask_rate, batch_size, train_epochs, learning_rate, dropout, temperature,
    extra: --use_amp for Traffic),
    finetune hyperparams (batch_size, learning_rate, dropout, head_dropout).
"""

import os
# 4090 portability: set SDSC_DATA_ROOT to point at a different machine's data dir.
# Default is the main 6000-Ada path; override with `export SDSC_DATA_ROOT=...`.
DATA_ROOT = os.environ.get("SDSC_DATA_ROOT", "/workspace/data/signal/forecasting")

DATASETS = {
    # ---------------------------------------------- ETT family
    "ETTh1": {
        "root_path": f"{DATA_ROOT}/ETT-small/",
        "data_path": "ETTh1.csv",
        "data": "ETTh1",
        "channels": 7,
        "pretrain": dict(seq_len=96, e_layers=3, n_heads=16, d_model=32, d_ff=64,
                         positive_nums=3, mask_rate=0.5, batch_size=32,
                         train_epochs=50, learning_rate=0.001, dropout=0.1),
        "finetune": dict(seq_len=96, label_len=48, pred_len=96,
                         batch_size=16, learning_rate=0.0001, dropout=0.2),
    },
    "ETTh2": {
        "root_path": f"{DATA_ROOT}/ETT-small/",
        "data_path": "ETTh2.csv",
        "data": "ETTh2",
        "channels": 7,
        "pretrain": dict(seq_len=96, e_layers=2, n_heads=8, d_model=8, d_ff=32,
                         positive_nums=3, mask_rate=0.5, batch_size=32,
                         train_epochs=50, learning_rate=0.001, dropout=0.1),
        "finetune": dict(seq_len=96, label_len=48, pred_len=96,
                         batch_size=16, learning_rate=0.0001, dropout=0.4,
                         head_dropout=0.2),
    },
    "ETTm1": {
        "root_path": f"{DATA_ROOT}/ETT-small/",
        "data_path": "ETTm1.csv",
        "data": "ETTm1",
        "channels": 7,
        "pretrain": dict(seq_len=96, e_layers=2, n_heads=8, d_model=32, d_ff=64,
                         positive_nums=3, mask_rate=0.5, batch_size=32,
                         train_epochs=50, learning_rate=0.001, dropout=0.1),
        "finetune": dict(seq_len=96, label_len=48, pred_len=96,
                         batch_size=16, learning_rate=0.0001, dropout=0.0),
    },
    "ETTm2": {
        "root_path": f"{DATA_ROOT}/ETT-small/",
        "data_path": "ETTm2.csv",
        "data": "ETTm2",
        "channels": 7,
        "pretrain": dict(seq_len=96, e_layers=3, n_heads=8, d_model=8, d_ff=16,
                         positive_nums=2, mask_rate=0.5, batch_size=32,
                         train_epochs=50, learning_rate=0.001, dropout=0.1),
        "finetune": dict(seq_len=96, label_len=48, pred_len=96,
                         batch_size=64, learning_rate=0.0001, dropout=0.0),
    },
    # ---------------------------------------------- Weather
    "Weather": {
        "root_path": f"{DATA_ROOT}/weather/",
        "data_path": "weather.csv",
        "data": "Weather",
        "channels": 21,
        "pretrain": dict(seq_len=96, e_layers=2, n_heads=8, d_model=64, d_ff=64,
                         positive_nums=2, mask_rate=0.5, batch_size=32,
                         train_epochs=50, learning_rate=0.001, dropout=0.1),
        "finetune": dict(seq_len=96, label_len=48, pred_len=96,
                         batch_size=16, learning_rate=0.0001, dropout=0.2),
    },
    # ---------------------------------------------- Electricity (321 channels)
    "ECL": {
        "root_path": f"{DATA_ROOT}/electricity/",
        "data_path": "electricity.csv",
        "data": "ECL",
        "channels": 321,
        # Original 3090 config (batch=16) was distributed over 4 GPUs via
        # --use_multi_gpu, so the per-GPU effective batch was ~4. On our single
        # 48GB 6000 Ada we mirror that effective batch: 4 + AMP. Encoder buffers
        # for SimMTM scale as B * (1 + positive_nums) * C * T * d_model;
        # ECL's C=321 multiplies the cost ~46× over ETTh1.
        "pretrain": dict(seq_len=96, label_len=48, e_layers=2, n_heads=16,
                         d_model=32, d_ff=64,
                         positive_nums=2, mask_rate=0.5, batch_size=4,
                         train_epochs=50, learning_rate=0.001, dropout=0.1,
                         temperature=0.02, use_amp=True),
        "finetune": dict(seq_len=96, label_len=48, pred_len=96,
                         batch_size=8, learning_rate=0.0001, dropout=0.2,
                         use_amp=True),
    },
    # ---------------------------------------------- Traffic (862 channels, heaviest)
    "Traffic": {
        "root_path": f"{DATA_ROOT}/traffic/",
        "data_path": "traffic.csv",
        "data": "Traffic",
        "channels": 862,
        # 4-GPU per-GPU effective batch was 1; we use 2 on single 48GB + AMP.
        # iTransformer SKIPPED (Pooler_Head 6B params).
        "pretrain": dict(seq_len=96, label_len=48, e_layers=3, n_heads=16,
                         d_model=128, d_ff=128,
                         positive_nums=2, mask_rate=0.5, batch_size=2,
                         train_epochs=50, learning_rate=0.001, dropout=0.2,
                         temperature=0.02, use_amp=True),
        "finetune": dict(seq_len=96, label_len=48, pred_len=96,
                         batch_size=4, learning_rate=0.0001, dropout=0.2,
                         use_amp=True),
    },
}

LOSS_MODES = ["mse", "sdsc", "hybrid", "dtw", "pcc", "snr", "zcr", "dilate"]
# User-requested 8 losses: MSE, SDSC, Hybrid, SoftDTW(=dtw), PCC, SI-SNR(=snr), ZCR, DILATE.
# MAE is still computed per epoch as a free side-metric, but not a training target.

# Models to sweep (C1 only — C2 deferred to next session pending external repo integration).
MODELS = ["SimMTM", "PatchTST", "iTransformer"]


# Architecture keys that MUST match between pretrain and finetune.
# If missing from finetune dict, they are inherited from pretrain dict.
_ARCH_KEYS = {"e_layers", "n_heads", "d_model", "d_ff"}


def emit_run_args(dataset_key, phase, model, loss_mode, seed):
    """Produce a list of `--flag value` strings to append to ``run.py``.

    phase: "pretrain" or "finetune"

    For finetune, architecture keys (e_layers, n_heads, d_model, d_ff) are
    inherited from the pretrain config so the model shape matches the saved
    checkpoint. Without this, argparse defaults (d_model=512, d_ff=2048, ...)
    create a shape mismatch and the pretrained weights are silently discarded.
    """
    d = DATASETS[dataset_key]
    p = dict(d[phase])  # copy so we don't mutate the original

    # Inherit architecture keys from pretrain into finetune if missing.
    if phase == "finetune":
        for ak in _ARCH_KEYS:
            if ak not in p and ak in d["pretrain"]:
                p[ak] = d["pretrain"][ak]

    args = [
        "--task_name", phase,
        "--root_path", d["root_path"],
        "--data_path", d["data_path"],
        "--model_id", d["data"],
        "--model", model,
        "--data", d["data"],
        "--features", "M",
        "--enc_in", str(d["channels"]),
        "--dec_in", str(d["channels"]),
        "--c_out", str(d["channels"]),
        "--loss_mode", loss_mode,
        "--seed", str(seed),
    ]
    for k, v in p.items():
        if k == "use_amp" and v:
            args.append("--use_amp")
            continue
        args.extend([f"--{k}", str(v)])
    if phase == "finetune":
        args.extend(["--is_training", "1"])
    return args


if __name__ == "__main__":
    # Quick dump for human inspection.
    import json
    for k in DATASETS:
        print(f"=== {k} ===")
        print(json.dumps(DATASETS[k], indent=2))
        print()
    print("Loss modes:", LOSS_MODES)
    print("Models:", MODELS)
    print("Total runs (one seed):",
          len(DATASETS) * len(LOSS_MODES) * len(MODELS))
