"""ECG dataset configuration for AAAI27 Classification Confirmatory.

ECG: 1 channel (lead II), T=1500, 4 arrhythmia classes (PhysioNet-derived).
Used primarily as a cross-domain pretrain source (ECG → Epilepsy) to
provide cardiac-vs-EEG modality diversity per AC-CL-7.
"""


class Config(object):
    def __init__(self):
        self.input_channels = 1
        self.increased_dim = 1
        self.final_out_channels = 128
        self.num_classes = 4
        self.num_classes_target = 4
        self.dropout = 0.35
        self.masking_ratio = 0.5
        self.lm = 3

        self.kernel_size = 25                   # longer T = larger receptive field
        self.stride = 3
        self.features_len = 162                 # ceil((1500 - 25) / 3) + 1 ≈ 162
        self.features_len_f = self.features_len

        self.num_epoch = 40
        self.TSlength_aligned = 1500
        self.CNNoutput_channel = 10

        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4
        self.lr_f = 3e-4

        self.drop_last = True
        self.batch_size = 16                    # longer T → smaller batch for memory
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
