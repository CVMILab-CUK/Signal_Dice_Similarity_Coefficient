"""Gesture dataset configuration for AAAI27 Classification Confirmatory.

Mirrors Epilepsy_Configs.py shape; only dataset-specific values differ.
Gesture: 3 channels (accelerometer x/y/z), T=206, 8 classes.
Source: TF-C benchmark (Zhang et al., NeurIPS 2022).
"""


class Config(object):
    def __init__(self):
        # signal shape
        self.input_channels = 3                # x, y, z accelerometer
        self.increased_dim = 1
        self.final_out_channels = 128
        self.num_classes = 8                    # 8 gesture types
        self.num_classes_target = 8
        self.dropout = 0.35
        self.masking_ratio = 0.5
        self.lm = 3                             # avg masking subseq length

        # encoder geometry
        self.kernel_size = 8
        self.stride = 1
        self.features_len = 24
        self.features_len_f = self.features_len

        # training
        self.num_epoch = 40
        self.TSlength_aligned = 206             # raw T for Gesture
        self.CNNoutput_channel = 32

        # optimizer
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4
        self.lr_f = 3e-4

        # data
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
