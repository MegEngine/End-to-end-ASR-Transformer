import os

# Audio:
num_mels = 80
num_freq = 1025
n_fft = 2 * (num_freq - 1)
sample_rate = 16000
frame_length_ms = 50
frame_shift_ms = 12.5
preemphasis = 0.97
min_level_db = -100
ref_level_db = 20

# Training:
dataset_root = "/data/asr_data/asr-mandarin/origin_data"
epochs = 200
max_steps = 600000
batch_size = 16  # 64 * 4
lr = 0.001

save_interval = 5000
log_interval = 100
checkpoint_path = "/data/asr_data/asr.codebase"

# # Model
eunits = 2048
elayers = 6
dunits = 2048
dlayers = 6
adim = 256
aheads = 4
dropout_rate = 0.1


# label smoothing
lsm_weight = 0.1

# recognize
beam_size = 15
penalty = 0.0
maxlenratio = 0.0
minlenratio = 0.0
