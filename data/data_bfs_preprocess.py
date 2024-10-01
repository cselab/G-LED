import numpy as np
import pdb
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class bfs_dataset(Dataset):
	def __init__(self,
				 data_location=['./data0.npy', './data1.npy'],
				 trajec_max_len=50,
				 start_n=0,
				 n_span=510):
		assert n_span > trajec_max_len
		self.start_n = start_n
		self.n_span  = n_span
		self.trajec_max_len = trajec_max_len

		solution0 = np.load(data_location[0],allow_pickle = True)
		solution1 = np.load(data_location[1],allow_pickle = True)
		solution  = np.concatenate([solution0,
									solution1],axis = 0)
		self.solution = torch.from_numpy(solution[start_n:start_n+n_span])

	def __len__(self):
		return self.n_span - self.trajec_max_len
		
	def __getitem__(self, index):
		return self.solution[index:index+self.trajec_max_len]


if __name__ == '__main__':
	dset = bfs_dataset()
	dloader = DataLoader(dataset=dset, batch_size = 20,shuffle = True)
	for iteration, batch in enumerate(dloader):
		print(iteration)
		print('Do something!')
		pdb.set_trace()