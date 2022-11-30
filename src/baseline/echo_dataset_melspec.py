import os
from pathlib import Path
import librosa
import torch
import random
import math
import numpy as np

from torch.utils.data import Dataset
from sklearn import preprocessing

import baseline_config

class EchoDatasetMelspec(Dataset):

    def __init__(self, dataset_root):
        
        # track where this dataset is from
        self.dataset_root = dataset_root
        
        # track total number of training samples
        self.total_size = 0
        
        # load the dataset filenames into RAM
        self.audio_dataset = self.from_dir_structure(dataset_root)
        
        # shuffle the dataset entries
        self.shuffle_dataset()
        
        if not os.path.exists('melspecs/'):
            os.makedirs('melspecs/')
              
    def from_dir_structure(self, dataset_path):
        
        # each sub directory represents a data class
        subfolders = [f for f in os.scandir(dataset_path) if f.is_dir()]

        # list of samples
        audio_dataset = []
        
        # perform a sparse encoding of the target labels
        le = preprocessing.LabelEncoder()
        targets = le.fit([folder.name for folder in subfolders])

        # load all files from each subfolder
        for subfolder in subfolders:
            
            # now get all the files in the folder
            audiofiles = [f for f in os.scandir(subfolder.path) if f.is_file()]

            for audiofile in audiofiles:
                
                # get metadata
                duration = librosa.get_duration(filename=audiofile.path)
                
                # transform category to sparse class target number
                target = le.transform([subfolder.name])
                
                # convert to tensor type the entropy loss function expects
                target = torch.tensor(target[0], dtype=torch.int64)

                # record the path and target
                audio_dataset.append((audiofile.path, subfolder.name, duration, target))

        # store the total length
        self.total_size = len(audio_dataset)
        
        return audio_dataset            

    # this shuffles the whole list of training samples
    def shuffle_dataset(self):
        random.shuffle(self.audio_dataset)

    # get sample at location 'index'
    def __getitem__(self, index):
        """Load waveform and target of an audio clip.
        Args:
            index: the index number
        Return: {
            "filename": str,
            "waveform": (clip_samples,),
            "target": (classes_num,)
        }
        """
        
        # retrieve the sample from the dataset
        sample = self.audio_dataset[index]
        
        # retrieve the duration
        duration = sample[2]
        
        waveform=None
        melspec =None
        
        # random offset within the audio file
        step=int(3)
        if duration > baseline_config.CLIP_LENGTH:
            offset = random.randint(0, int(math.floor(duration-baseline_config.CLIP_LENGTH))) // step
            offset = offset * step
        else:
            offset = 0
          
        # print("duration",duration)    
        # print("offset",offset)
        
        # generate melspectrogram
        melspec_file = 'melspecs/' + Path(sample[0]).name + "-" + sample[1]+ "-" + str(offset) + '.mel'
        if not os.path.exists(melspec_file):
            
            # load the waveform
            waveform, sr = librosa.load(sample[0], 
                                    sr = baseline_config.sample_rate, 
                                    duration = min(duration, baseline_config.CLIP_LENGTH),
                                    offset=offset, 
                                    mono=True)
        
            # pad the waveform if it is too short
            if duration < baseline_config.CLIP_LENGTH:
                waveform = librosa.util.pad_center(waveform, 
                                                   size=baseline_config.CLIP_LENGTH*baseline_config.sample_rate, 
                                                   mode='constant')
       
            melspec = librosa.feature.melspectrogram(waveform, 
                                                sr=baseline_config.sample_rate, 
                                                S=None, 
                                                n_fft=baseline_config.window_size, 
                                                hop_length=baseline_config.hop_size, 
                                                win_length=None, 
                                                window='hann', 
                                                center=True, 
                                                n_mels=baseline_config.mel_bins,
                                                pad_mode='constant',
                                                fmin=baseline_config.fmin,
                                                fmax=baseline_config.fmax,
                                                power=2.0)
            melspec = np.expand_dims(melspec, 0)     # C F T 
            melspec = np.transpose(melspec, (0,2,1)) # C T F
            np.save(melspec_file, melspec)
        else:
            melspec = np.load(melspec_file)
            
        # print("melspec shape",melspec.shape)    
                
        # return a dictionary with the sample data
        return {
            "filename": sample[0],
            "waveform": waveform,
            "melspec": melspec,
            "target": sample[3],
        }

    def __len__(self):
        return self.total_size
    
