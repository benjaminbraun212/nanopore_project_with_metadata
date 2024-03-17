import torch
from random import randint
import numpy as np
import pandas as pd
from config import *
#from scipy.signal import resample
from torchvision import transforms
import pickle
import os
import random
import h5py
from torch.utils.data import Dataset, ConcatDataset
from fastai.tabular.core import *
from fastai.tabular.all import *
from fastai.data.load import _FakeLoader, _loaders
from fastai.vision.all import *

METADATA_FILENAME = 'signal_mapping_tuple.pkl'

class FourierTransform(torch.nn.Module):
    def __init__(self, sampling_size):
        super().__init__()
        self.sampling_size = sampling_size

    def forward(self, signal):
        return resample(signal, self.sampling_size)


class differences_transform(torch.nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self, signal):
		return np.diff(signal)


class startMove_transform(torch.nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self, signal):
		startPosModification = randint(0, 60)
		if len(signal[startPosModification: ]) < 4000:
			num_missing_values = 4000 - len(signal[startPosModification: ])
			return np.pad(signal[startPosModification: ], (0, num_missing_values), 'constant')
		return signal[startPosModification: ]
	
class truncate_transform(torch.nn.Module):
	def __init__(self, truncate_limit):
		super().__init__()
		self.truncate_limit = truncate_limit
	def forward(self, signal):
		if len(signal) == self.truncate_limit:
			return signal
		if len(signal) > self.truncate_limit:
			return signal[:self.truncate_limit]
		else:
			num_missing_values = self.truncate_limit - len(signal)
			return signal + [0] * num_missing_values

class cutToWindows_transform(torch.nn.Module):
	def __init__(self, seqLength, stride, winLength):
		super().__init__()
		self.seqLength = seqLength
		self.stride = stride
		self.winLength = winLength
	def forward(self, signal):
		splitInput = np.zeros((self.seqLength, self.winLength), dtype="int16")
		for i in range(self.seqLength):
			# print(len(signal), signal)
			splitInput[i, :] = signal[(i*self.stride):(i*self.stride)+self.winLength]
		return splitInput


class noise_transform(torch.nn.Module):
	def __init__(self, noise_range):
		super().__init__()
		self.noise_range = noise_range
	def forward(self, signal):
		shape = tuple(signal.shape)
		noise = np.random.normal(0,self.noise_range, size = shape)
		if isinstance(signal, np.ndarray):
			return signal + noise
		else:
			noise = torch.from_numpy(noise).float().to(device=signal.device)
			return signal + noise


class NoisyDataset(Dataset):
    def __init__(self, original_dataset, noise_std=0.15, num_samples=2):
        self.original_dataset = original_dataset
        self.noise_std = noise_std
        self.num_samples = num_samples

    def __len__(self):
        return len(self.original_dataset) * (self.num_samples + 1)  # +1 for the original dataset

    def __getitem__(self, idx):
        if idx % (self.num_samples + 1) == 0:
            # Return an item from the original dataset
            return self.original_dataset[idx // (self.num_samples + 1)]
        else:
            # Generate a noisy sample
            original_idx = idx // (self.num_samples + 1)
            original_data, original_target = self.original_dataset[original_idx]

            # Add noise to the original data
            noisy_data = original_data + torch.randn_like(original_data) * self.noise_std

            return noisy_data, original_target
		
def load_multiple_ds_metadata(path_list, limit):
	meta_list = []
	for path in path_list:
		meta = _load_metadata_pickle(os.path.join(path, METADATA_FILENAME))
		meta = meta.values.tolist()
		random.shuffle(meta)
		meta_list.extend(meta[0:min(limit, len(meta))])
	return meta_list

def _load_metadata_pickle(path):
	reads_metadata = pd.read_pickle(path) 
	return reads_metadata


def create_labels_df_from_data_dicts(data_dict_list):
	overall_df = pd.DataFrame()
	for data_dict, train_test_flag in data_dict_list:
		for label in [0,1]:
			df_tmp = pd.DataFrame()
			df_tmp["read_id"] = [read_id for path, read_id, *junk in data_dict[label]]
			df_tmp["path"] = [path for path, read_id, *junk in data_dict[label]]
			df_tmp["label"] = label
			df_tmp["train_test_flag"] = train_test_flag
			overall_df = pd.concat([overall_df,df_tmp])
	return overall_df


def get_signal_start_index(signal_array):
    start_point_idx, start_point_val = signal_array[10:3000].argmax(), signal_array[10:3000].max()
    pre_start = signal_array[start_point_idx - 19:start_point_idx - 1]
    pro_start = signal_array[start_point_idx + 1:start_point_idx + 19]
    if len(pro_start) == 0 or len(pre_start) == 0:
        return 0
    if start_point_val > pre_start.mean():
        if start_point_val > pro_start.mean():
            if np.var(signal_array[start_point_idx + 50:start_point_idx + 80]) > np.var(
                    signal_array[start_point_idx - 80:start_point_idx - 50]):
                return start_point_idx+600
    ## if couldnt find valid start poin then return 0 as start point
    return 0

def get_signal_array(path, read_id):
	try:
		with h5py.File(path, 'r') as f:
			if 'Raw' in f:  # Single read file
				signal = f.get('Raw/Reads/read_' + read_id + '/Signal')
			else:  # Multi read file
				signal = f.get('read_'+read_id).get('Raw/Signal')
			return signal[:]
	except (AttributeError, FileNotFoundError) as e:
		path = path.replace("_pass", "_fail")
		with h5py.File(path, 'r') as f:
			if 'Raw' in f:  # Single read file
				signal = f.get('Raw/Reads/' + read_id + '/Signal')
			else:  # Multi read file
				signal = f.get('read_'+read_id).get('Raw/Signal')
			return signal[:]

def get_tabular_model(to, cont_names, out_sz, layers, embed_p, ps):
    return TabularModel(emb_szs = get_emb_sz(to),
                        n_cont = len(cont_names),
                        out_sz = out_sz,
                        layers = layers,
						embed_p=embed_p,
						ps=ps
						)