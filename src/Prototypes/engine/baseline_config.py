#
# Configuration file adapted for Project Echo
#


# ############################################################################################
# General parameters used by Project Echo
# ############################################################################################
SAMPLE_RATE   = 44100//2 # all the samples are converted to bit rate of 32050 (Samples/Second)
MIN_FREQUENCY = 0        # minimum frequency (Hz) for the Fast Fourier Transform related functions
MAX_FREQUENCY = 44100//4 # maximum frequency (Hz) for the Fast Fourier Transform related functions
HOP_LENGTH    = 512      # the number of samples to slide spectrogram window along the audio samples
NUMBER_FFT    = 512      # the number of FFT to execute within a single spectrogram window
NUMBER_MELS   = 128      # the number of Mel-Spectrogram groups to split the frequency dimension
CLIP_LENGTH   = 5        # only look at this many seconds of clip randomly within the audio file

CLASSES_NUM   = 20       # this is the number of animal species we have in the dataset


# ############################################################################################
# Adapted configuration parameters below
# ############################################################################################
exp_name          = "project_echo" # the saved ckpt prefix name of the model 
workspace         = "./"           # the folder of your code
dataset_path      = "d:/data/combined/OUTPUT_tensors/" # the dataset path
classes_num       = CLASSES_NUM

loss_type         = "clip_ce" # cross entropy loss

# enable debugging to assist with integration
debug             = False

# some defaults - these may be overridden by the code
random_seed       = 970131  # 19970318 970131 12412 127777 1009 34047
batch_size        = 32      # default is 32
learning_rate     = 1e-4    # 1e-4 also workable 
max_epoch         = 70      # early stop
num_workers       = 0       # change to >= 1 for multi-threaded (needs fixing)

# scheduling curve (warm start for transformer based model)
lr_scheduler_epoch = [10,20,30]
lr_rate            = [0.02, 0.05, 0.1]

# these data preparation optimizations do not bring many improvements, so deprecated
enable_repeat_mode = False # repeat the spectrogram / reshape the spectrogram

# for model's design
enable_tscam = False # enbale the token-semantic layer


# ############################################################################################
# signal processing hyperparamaters
# ############################################################################################
sample_rate  = SAMPLE_RATE               # default: 32000 
clip_samples = sample_rate * CLIP_LENGTH # default: 10 seconds worth
window_size  = NUMBER_FFT                # default: set window size same as FFT count
hop_size     = HOP_LENGTH                # default: 320 works with 10 seconds, 32000 bitrate
mel_bins     = NUMBER_MELS               # default: 64 number of frequency bands
fmin         = MIN_FREQUENCY             # default: 5
fmax         = MAX_FREQUENCY             # default: 14000
shift_max    = int(clip_samples * 0.5)


# ############################################################################################
# htsat hyperparamaters
# ############################################################################################
htsat_weight_decay = 0.025       # default: 0.05
htsat_window_size  = 8           # default: 8 
htsat_spec_size    = 512         # default: 256
htsat_patch_size   = 4           # default: 4 
htsat_stride       = (4, 4)      # default: (4, 4)
htsat_num_head     = [4,8,16,32] # default: [4,8,16,32]
htsat_dim          = 96          # default: 96 
htsat_depth        = [2,2,6,2]   # default: [2,2,6,2]

