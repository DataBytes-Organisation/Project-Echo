#
# Configuration file adapted for Project Echo
#


# ############################################################################################
# General parameters used by Project Echo
# ############################################################################################
SAMPLE_RATE   = 32000   # all the samples are converted to bit rate of 32000 (Samples/Second)
MIN_FREQUENCY = 16      # minimum frequency (Hz) for the Fast Fourier Transform related functions
MAX_FREQUENCY = 4096*3  # maximum frequency (Hz) for the Fast Fourier Transform related functions
HOP_LENGTH    = 256     # the number of samples to slide spectrogram window along the audio samples
NUMBER_FFT    = 1024    # the number of FFT to execute within a single spectrogram window
NUMBER_MELS   = 128     # the number of Mel-Spectrogram groups to split the frequency dimension
CLIP_LENGTH   = 8       # only look at this many seconds of clip randomly within the audio file


# ############################################################################################
# Adapted configuration parameters below
# ############################################################################################
exp_name          = "project_echo" # the saved ckpt prefix name of the model 
workspace         = "./"           # the folder of your code
dataset_path      = "C:/Users/Andrew/OneDrive - Deakin University/DataSets/birdclef2022/" # the dataset path

# AudioSet & SCV2: "clip_bce" |  ESC-50: "clip_ce" 
loss_type         = "clip_ce" # cross entropy loss  ("clip_ce" or "clip_bce")

# trained from a checkpoint, or evaluate a single model 
#resume_checkpoint = None # workspace + "/ckpt/htsat_audioset_pretrain.ckpt"
esc_fold          = 0 # just for esc dataset, select the fold you need for evaluation and (+1) validation

# enable debugging to assist with integration
debug             = True

random_seed       = 970131 # 19970318 970131 12412 127777 1009 34047
batch_size        = 128 # batch size per GPU x GPU number , default is 32 x 4 = 128
learning_rate     = 1e-3 # 1e-4 also workable 
max_epoch         = 500
num_workers       = 0 # change to >= 1 for multi-threaded

# scheduling curve (warm start for transformer based model)
lr_scheduler_epoch = [10,20,30]
lr_rate            = [0.02, 0.05, 0.1]

# these data preparation optimizations do not bring many improvements, so deprecated
#enable_token_label = False # token label
#class_map_path = "class_hier_map.npy"
#class_filter = None 
#retrieval_index = [15382, 9202, 130, 17618, 17157, 17516, 16356, 6165, 13992, 9238, 5550, 5733, 1914, 1600, 3450, 13735, 11108, 3762, 
#    9840, 11318, 8131, 4429, 16748, 4992, 16783, 12691, 4945, 8779, 2805, 9418, 2797, 14357, 5603, 212, 3852, 12666, 1338, 10269, 2388, 8260, 4293, 14454, 7677, 11253, 5060, 14938, 8840, 4542, 2627, 16336, 8992, 15496, 11140, 446, 6126, 10691, 8624, 10127, 9068, 16710, 10155, 14358, 7567, 5695, 2354, 8057, 17635, 133, 16183, 14535, 7248, 4560, 14429, 2463, 10773, 113, 2462, 9223, 4929, 14274, 4716, 17307, 4617, 2132, 11083, 1039, 1403, 9621, 13936, 2229, 2875, 17840, 9359, 13311, 9790, 13288, 4750, 17052, 8260, 14900]
#token_label_range = [0.2,0.6]
#enable_time_shift = False # shift time
#enable_label_enhance = False # enhance hierarchical label
enable_repeat_mode = False # repeat the spectrogram / reshape the spectrogram

# for model's design
enable_tscam = False # enbale the token-semantic layer

# for signal processing
sample_rate = SAMPLE_RATE # 32000 # 16000 for scv2, 32000 for audioset and esc-50
clip_samples = sample_rate * CLIP_LENGTH # audio_set 10-sec clip
window_size = NUMBER_FFT
hop_size = HOP_LENGTH # 320 # 160 for scv2, 320 for audioset and esc-50
mel_bins = NUMBER_MELS # 64
fmin = MIN_FREQUENCY # 5
fmax = MAX_FREQUENCY # 14000
shift_max = int(clip_samples * 0.5)

# for data collection
classes_num = 5
#patch_size = (25, 4) # deprecated
crop_size = None # int(clip_samples * 0.5) deprecated

# for htsat hyperparamater
htsat_window_size = 8   # 8 
htsat_spec_size =  256  # 256
htsat_patch_size = 4    # 4 
htsat_stride = (4, 4)
htsat_num_head = [4,8,16,32]
htsat_dim = 96          # 96 
htsat_depth = [2,2,6,2]

# swin_pretrain_path = None
# "/home/Research/model_backup/pretrain/swin_tiny_c24_patch4_window8_256.pth"

# Some Deprecated Optimization in the model design, check the model code for details
htsat_attn_heatmap = False
htsat_hier_output = False 
htsat_use_max = False


