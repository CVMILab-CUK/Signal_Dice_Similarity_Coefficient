#!/bin/bash
# AAAI27 Plan C-4+C-5 backbone setup. Clones TF-C + TS2Vec, applies our
# config patches, creates dataset symlinks. Idempotent — safe to re-run.
#
# Run from repo root: bash paper_supplement/scripts/setup_backbones.sh

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$REPO_ROOT/backbones"
cd "$REPO_ROOT/backbones"

# ── 1. Clone TF-C original ───────────────────────────────────────────────
if [ ! -d "TFC" ]; then
  echo "[setup] cloning TF-C upstream..."
  git clone --depth=1 https://github.com/mims-harvard/TFC-pretraining.git TFC
fi

# ── 2. Clone TS2Vec original ────────────────────────────────────────────
if [ ! -d "TS2Vec" ]; then
  echo "[setup] cloning TS2Vec upstream..."
  git clone --depth=1 https://github.com/yuezhihan/ts2vec.git TS2Vec
fi

# ── 3. Apply TF-C config patches (upstream had bugs / missing files) ─────
TFC_CONFIGS="TFC/code/config_files"

# 3a. Add missing TSlength_aligned=178 to Epilepsy_Configs.py
if ! grep -q "TSlength_aligned" "$TFC_CONFIGS/Epilepsy_Configs.py"; then
  echo "[setup] patching TFC Epilepsy_Configs.py (add TSlength_aligned=178)"
  /usr/bin/python3 - <<'PY'
import re, pathlib
p = pathlib.Path("TFC/code/config_files/Epilepsy_Configs.py")
src = p.read_text()
patched = re.sub(
    r"(self\.features_len_f = 24[^\n]*\n)",
    r"\1\n        # AAAI27 protocol v2: upstream config omitted TSlength_aligned.\n"
    r"        # Epilepsy data is (60, 1, 178) — T=178.\n"
    r"        self.TSlength_aligned = 178\n",
    src,
    count=1,
)
p.write_text(patched)
PY
fi

# 3b. Correct HAR_Configs.py TSlength_aligned from 206 to 128 (our data T=128)
if grep -q "self.TSlength_aligned = 206" "$TFC_CONFIGS/HAR_Configs.py"; then
  echo "[setup] patching TFC HAR_Configs.py (TSlength_aligned 206 -> 128)"
  sed -i \
    's/self\.TSlength_aligned = 206/# AAAI27 protocol v2: HAR data is (N, 9, 128) - T=128 (not upstream 206).\n        self.TSlength_aligned = 128/' \
    "$TFC_CONFIGS/HAR_Configs.py"
fi

# 3c. Add Gesture_Configs.py (upstream missing)
if [ ! -f "$TFC_CONFIGS/Gesture_Configs.py" ]; then
  echo "[setup] adding TFC Gesture_Configs.py (upstream missing)"
  cat > "$TFC_CONFIGS/Gesture_Configs.py" <<'CFG'
"""TF-C Gesture config — added for AAAI27 protocol v2 cross-backbone study.
Gesture data: (320, 3, 206) — projected to 1 channel per TF-C convention.
"""


class Config(object):
    def __init__(self):
        self.input_channels = 1
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 32
        self.TSlength_aligned = 206
        self.num_classes = 8
        self.num_classes_target = 8
        self.dropout = 0.35
        self.features_len = 24
        self.features_len_f = 24
        self.num_epoch = 40
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4
        self.lr_f = 3e-4
        self.drop_last = True
        self.batch_size = 32
        self.target_batch_size = 16
        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 0.001
        self.jitter_ratio = 0.001
        self.max_seg = 5


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True
        self.use_cosine_similarity_f = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 10
CFG
fi

# ── 4. Dataset symlinks (TF-C expects ../../datasets/{name}) ─────────────
mkdir -p TFC/datasets
for ds in Epilepsy SleepEEG Gesture HAR ECG; do
  src="/workspace/data/signal/classification/$ds"
  dst="TFC/datasets/$ds"
  if [ ! -e "$dst" ] && [ -d "$src" ]; then
    ln -s "$src" "$dst"
    echo "[setup] linked TFC/datasets/$ds"
  fi
done

# ── 5. Verify TF-C configs load ─────────────────────────────────────────
echo "[setup] verifying TF-C configs..."
cd TFC/code
/usr/bin/python3 -c "
import sys; sys.path.insert(0, '.')
for ds in ['Epilepsy','Gesture','HAR','ECG','SleepEEG']:
    m = __import__(f'config_files.{ds}_Configs', fromlist=['Config'])
    c = m.Config()
    assert hasattr(c, 'TSlength_aligned'), ds
    print(f'  {ds:10s}: T={c.TSlength_aligned} classes={c.num_classes}')
"
echo "[setup] backbone setup complete."
