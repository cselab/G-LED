import argparse
import pdb
import os
import sys
from torch.utils.data import DataLoader
import torch

from mimagen_pytorch import Unet3D, ElucidatedImagen, ImagenTrainer
from main_seq_bfs import Args as Args_seq
from main_diff_bfs import Args as Args_diff
sys.path.insert(0, './util')
from utils import read_args_txt
sys.path.insert(0, './data')
from data_bfs_preprocess import bfs_dataset 
sys.path.insert(0, './transformer')
from sequentialModel import SequentialModel as transformer
sys.path.insert(0, './train_test_spatial')
from test_diff import test_final_overall 
from test_diff_ensamble import test_final_overall_ensamble



class Args_final_eval:
	def __init__(self):
		self.parser = argparse.ArgumentParser()
		"""
		for finding the dynamics dir
		"""
		self.parser.add_argument("--bfs_dynamic_folder", 
								 default='output/bfs_les_2023_12_21_12_09_10',
								 help='all the information of bfs training')
		
		"""
		reading the seq model
		"""
		self.parser.add_argument("--Nt_read",
								 default = 40,
								 help = "Which Nt model we need to read")
		self.parser.add_argument("--use_best",
								 default = True)
		"""
		reading the diffusion model
		"""
		self.parser.add_argument("--Nepoch_read",
								 default = 2,
								 help = "Which epoch model we need to read")

		"""
		for dataset
		"""
		self.parser.add_argument("--trajec_max_len", 
								 default=151,
								 help = 'max seq_length (per seq) to test the model')
		self.parser.add_argument("--start_n", 
								 default=9500, 
								 help = 'the starting step of the data')
		self.parser.add_argument("--n_span",
								 default=152,
								 help='the total step of the data from the staring step')

		"""
		for seq_net_eval
		"""
		self.parser.add_argument("--test_Nt", 
								 default=150,
								 help = 'How many step you want to proceed! Should be divided by 10')
		


		
		"""
		for eval dataset hyperparameter
		"""
		self.parser.add_argument("--batch_size", default = 1)
		self.parser.add_argument("--device", type=str, default = "cuda:0")
		


	def update_args(self):
		args = self.parser.parse_args()
		args.seq_args_txt  = os.path.join(args.bfs_dynamic_folder,
										 'logging','args.txt' )
		args.diff_args_txt = os.path.join(args.bfs_dynamic_folder,
										 'diffusion_folder',
										 'logging','args.txt')
		
		# output dataset
		args.experiment_path = os.path.join(args.bfs_dynamic_folder,
											'diffusion_folder',
											'experiment_final')
		if not os.path.isdir(args.experiment_path):
			os.makedirs(args.experiment_path)
		return args

if __name__ == '__main__':
	"""
	Fetch args
	"""
	args_final = Args_final_eval()
	args_final = args_final.update_args()	
	args_seq  = read_args_txt(Args_seq(), 
							  args_final.seq_args_txt)
	args_diff = read_args_txt(Args_diff(), 
							  args_final.diff_args_txt)
	
	"""
	Fetch dataset
	"""
	data_set = bfs_dataset(data_location  = args_seq.data_location,
						   trajec_max_len = args_final.trajec_max_len,
						   start_n        = args_final.start_n,
						   n_span         = args_final.n_span)
	
	data_loader = DataLoader(dataset=data_set, 
							 shuffle=False,
							 batch_size=args_final.batch_size)

	
	
	"""
	Fetch models
	"""
	model = transformer(args_seq).to(args_final.device).float()
	print('Number of parameters: {}'.format(model._num_parameters()))
	if args_final.use_best:
		model.load_state_dict(torch.load(args_seq.current_model_save_path+'best_model_sofar'))
	else:
		model.load_state_dict(torch.load(args_seq.current_model_save_path+'model_epoch_'+str(args_final.Nt_read),map_location=torch.device(args_final.device)))	
	
	unet1 = Unet3D(dim=args_diff.unet_dim,
				   cond_images_channels=2, 
				   memory_efficient=True, 
				   dim_mults=(1, 2, 4, 8)).to(torch.device(args_diff.device))  #mid: mid channel
	image_sizes = (512)
	image_width = (512) 
	imagen = ElucidatedImagen(
            unets = (unet1),
            image_sizes = image_sizes,
            image_width = image_width,   
            channels = 2,   # Han Gao add the input to this args explicity     
            random_crop_sizes = None,
            num_sample_steps = args_diff.num_sample_steps, # original is 10
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
            ).to(torch.device(args_final.device))
	trainer = ImagenTrainer(imagen, device =torch.device(args_final.device))
	trainer.load(path=args_diff.model_save_path+'/best_model_sofar')
	test_final_overall_ensamble(args_final, 
					   args_seq, 
					   args_diff, 
					   trainer, 
					   model,
					   data_loader)
	exit()
	test_final_overall(args_final, 
					   args_seq, 
					   args_diff, 
					   trainer, 
					   model,
					   data_loader)








