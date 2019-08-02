# Audio
num_mels = 80
num_freq = 1025
sample_rate = 20000
frame_length_ms = 50
frame_shift_ms = 12.5
tisv_frame = 180
preemphasis = 0.97
min_level_db = -100
ref_level_db = 20
griffin_lim_iters = 60
power = 1.5

# Model
n_mels_channel = 80
hidden_dim = 256
num_layer = 3
speaker_dim = 32
class_num = 108
re_num = 1e-6

# Train
dataset_path = "./dataset"
origin_data = "./VCTK-Corpus-Processed"
N = 16
M = 10
learning_rate = 0.01
epochs = 10000
checkpoint_path = "./model_new"
save_step = 5000
log_step = 5
clear_Time = 20
