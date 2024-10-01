import sys
import argparse
import pdb
from datetime import datetime
import os
from torch.utils.data import DataLoader
import torch
import numpy as np
import json


"""
Internal pacakage
"""
from main_seq_bfs import Args as SEQ_ARGS
from mimagen_pytorch import Unet3D, ElucidatedImagen, ImagenTrainer

sys.path.insert(0, './util')
from utils import save_args, read_args_txt
sys.path.insert(0, './data')
from data_bfs_preprocess import bfs_dataset 
sys.path.insert(0, './train_test_spatial')
from  train_diff import train_diff

class Args:
	def __init__(self):
		self.parser = argparse.ArgumentParser()
		"""
		for finding the dynamics dir
		"""
		self.parser.add_argument("--bfs_dynamic_folder", 
								 default='output/bfs_les_2023_10_31_17_49_35',
								 help='all the information of ks training')
		"""
		for diffusion model
		"""
		self.parser.add_argument("--Nt",
								 default = 10,
								 help = 'Time steps we use as a single seq')
		self.parser.add_argument("--unet_dim", 
								 default=32,
								 help='The unet dimension')
		self.parser.add_argument("--num_sample_steps", 
								 default=20,
								 help='The noise forward/reverse step')
		
		"""
		for training 
		"""
		self.parser.add_argument("--batch_size", default = 1)
		self.parser.add_argument("--epoch_num", default = 20)
		self.parser.add_argument("--device", type=str, default = "cuda:1")
		self.parser.add_argument("--shuffle",default=True)
		


	def update_args(self):
		args = self.parser.parse_args()
		# output dataset
		args.experiment_path = os.path.join(args.bfs_dynamic_folder,'diffusion_folder')
		if not os.path.isdir(args.experiment_path):
			os.makedirs(args.experiment_path)
		args.model_save_path = os.path.join(args.experiment_path,'model_save')
		if not os.path.isdir(args.model_save_path):
			os.makedirs(args.model_save_path)
		args.logging_path = os.path.join( args.experiment_path,'logging') 
		if not os.path.isdir(args.logging_path):
			os.makedirs(args.logging_path)

		args.seq_args_txt = os.path.join(args.bfs_dynamic_folder,
										 'logging','args.txt' )
		return args

if __name__ == '__main__':

	"""
	Diff args
	"""
	diff_args = Args()
	diff_args = diff_args.update_args()
	save_args(diff_args)
	"""
	Sequence args
	"""
	seq_args = read_args_txt(SEQ_ARGS(),diff_args.seq_args_txt)
	
	"""
	Fetch dataset
	"""
	data_set = bfs_dataset(data_location  = seq_args.data_location,
						   trajec_max_len = diff_args.Nt,#seq_args.trajec_max_len,
						   start_n        = seq_args.start_n,
						   n_span         = seq_args.n_span)
	data_loader = DataLoader(dataset=data_set, 
							 shuffle=diff_args.shuffle,
							 batch_size=diff_args.batch_size)
	
	"""
	Create diffusion model
	"""
	unet1 = Unet3D(dim=diff_args.unet_dim,
				   cond_images_channels=2, 
				   memory_efficient=True, 
				   dim_mults=(1, 2, 4, 8)).to(torch.device(diff_args.device))  #mid: mid channel
	image_sizes = (512)
	image_width = (512)
	imagen = ElucidatedImagen(
		unets = (unet1),
		image_sizes = image_sizes,
		image_width = image_width,   
		channels = 2,   # Han Gao add the input to this args explicity     
		random_crop_sizes = None,
		num_sample_steps = diff_args.num_sample_steps, # original is 10
		cond_drop_prob = 0.1,
		sigma_min = 0.002,
		sigma_max = (80),      # max noise level, double the max noise level for upsampler  （80，160）
		sigma_data = 0.5,      # standard deviation of data distribution
		rho = 7,               # controls the sampling schedule
		P_mean = -1.2,         # mean of log-normal distribution from which noise is drawn for training
		P_std = 1.2,           # standard deviation of log-normal distribution from which noise is drawn for training
		S_churn = 80,          # parameters for stochastic sampling - depends on dataset, Table 5 in apper
		S_tmin = 0.05,
		S_tmax = 50,
		S_noise = 1.003,
		condition_on_text = False,
		auto_normalize_img = False  # Han Gao make it false
		).to(torch.device(diff_args.device))
	trainer = ImagenTrainer(imagen, device =torch.device(diff_args.device))
	train_diff(diff_args=diff_args,
               seq_args=seq_args,
               trainer=trainer,
               data_loader=data_loader)