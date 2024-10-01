from main_seq_bfs import Args
import torch
import sys
import argparse
from torch.utils.data import DataLoader
import pdb
# Internal package
sys.path.insert(0, './util')
from utils import read_args_txt
sys.path.insert(0, './train_test_seq')
#from test_seq import eval_seq_overall
from test_seq import eval_seq_overall
sys.path.insert(0, './data')
from data_bfs_preprocess import bfs_dataset 
sys.path.insert(0, './transformer')
from sequentialModel import SequentialModel as transformer

class Args_seq_sample:
	def __init__(self):
		self.parser = argparse.ArgumentParser()
		"""
		for training args txt
		"""
		self.parser.add_argument("--train_args_txt", 
		default = 'output/bfs_les_2023_10_31_17_49_35/logging/args.txt',
								help = 'load the args_train')
		self.parser.add_argument("--Nt_read",
								 default = 30,
								 help = "Which Nt model we need to read")

		
		
		"""
		for dataset
		"""
		self.parser.add_argument("--trajec_max_len", 
								 default=41,
								 help = 'max seq_length (per seq) to test the model')
		self.parser.add_argument("--start_n", 
								 default=8500,
								 help = 'the starting step of the data')
		self.parser.add_argument("--n_span",
								 default=42,
								 help='the total step of the data from the staring step')
			   

		
		"""
		for testing
		"""
		self.parser.add_argument("--test_Nt",
								 default=40,
								 help='The length of forward propgatate')
		self.parser.add_argument("--batch_size",
								 default=1,
								 help = 'how many seqs you want to test together per bp')
		self.parser.add_argument("--shuffle",
								 default=False,
								 help = 'shuffle the batch')
		self.parser.add_argument("--device",
								 default='cuda:0')
		
	
	def update_args(self):
		args = self.parser.parse_args()
		args.experiment_path = None
		return args
		
if __name__ == '__main__':
	"""
	Fetch args
	"""
	args_sample = Args_seq_sample()
	args_sample = args_sample.update_args()
	args_train = read_args_txt(Args(), 
							   args_sample.train_args_txt)
	args_train.device = args_sample.device
	args_sample.experiment_path = args_train.experiment_path
	
	"""
	Pre-check
	"""
	assert args_train.coarse_dim[0]*args_train.coarse_dim[1]*2 == args_train.n_embd

	"""
	Fetch dataset
	"""
	data_set = bfs_dataset(data_location  = args_train.data_location,
						   trajec_max_len = args_sample.trajec_max_len,
						   start_n        = args_sample.start_n,
						   n_span         = args_sample.n_span)
	data_loader = DataLoader(dataset=data_set, 
							 shuffle=args_sample.shuffle,
							 batch_size=args_sample.batch_size)

	"""
	Create and Load model
	"""
	model = transformer(args_train).to(args_sample.device).float()
	print('Number of parameters: {}'.format(model._num_parameters()))
	model.load_state_dict(torch.load(args_train.current_model_save_path+'model_epoch_'+str(args_sample.Nt_read), map_location=torch.device(args_sample.device)))

	"""
	create loss function
	"""
	loss_func = torch.nn.MSELoss()


	"""
	Eval
	"""
	eval_seq_overall(args_train=args_train,
					 args_sample=args_sample,
					 model=model, 
					 data_loader=data_loader, 
					 loss_func=loss_func)