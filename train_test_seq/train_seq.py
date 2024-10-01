import pdb
import torch
from tqdm import tqdm
from test_seq import test_epoch
import os
import time
import sys
sys.path.insert(0, './util')
from utils import save_model
def train_seq_shift(args, 
					model, 
					data_loader, 
					data_loader_copy,
					data_loader_valid,
					loss_func, 
					optimizer,
					scheduler):
	# N C H W
	down_sampler = torch.nn.Upsample(size=args.coarse_dim, 
								     mode=args.coarse_mode)
	Nt = args.start_Nt
	for epoch in tqdm(range(args.epoch_num)):
		tic = time.time()
		print('Start epoch '+ str(epoch)+' at Nt ', Nt)
		if epoch >0:
			max_mre,min_mre, mean_mre, sigma3 = test_epoch(args=args,
														   model=model, 
														   data_loader=data_loader_valid,
														   loss_func=loss_func,
														   Nt=Nt,
														   down_sampler=down_sampler,
														   ite_thold = 2)
			print('#### max  re valid####=',max_mre)
			print('#### mean re valid####=',mean_mre)
			print('#### min  re valid####=',min_mre)
			print('#### 3 sigma valid####=',sigma3)
			print('Last LR is '+str(scheduler.get_last_lr()))
			max_mre,min_mre, mean_mre, sigma3 = test_epoch(args = args,
							                               model = model, 
                                                           data_loader = data_loader_copy,
                                                           loss_func = loss_func,
                                                           Nt = Nt,
                                                           down_sampler = down_sampler,
                                                           ite_thold = 5)
			print('#### max  re train####=',max_mre)
			print('#### mean re train####=',mean_mre)
			print('#### min  re train####=',min_mre)
			print('#### 3 sigma train ####=',sigma3)
			if (max_mre < args.march_tol) or (mean_mre < args.march_tol*0.1):
				save_model(model, args, Nt, bestModel = True)
				Nt += args.d_Nt
				scheduler.step()
				continue
		
		model = train_epoch(args=args,
							model=model, 
							data_loader=data_loader,
							loss_func=loss_func,
							optimizer=optimizer,
							down_sampler=down_sampler)
				
		print('Epoch elapsed ', time.time()-tic)
	save_model(model, args, Nt, bestModel = False)
def train_epoch(args, 
				model, 
				data_loader, 
				loss_func, 
				optimizer,
				down_sampler):
	print('Nit = ',len(data_loader))
	for iteration, batch in tqdm(enumerate(data_loader)):	
		batch = batch.to(args.device).float()

		b_size = batch.shape[0]
		num_time = batch.shape[1]
		num_velocity = 2
		batch = batch.reshape([b_size*num_time, num_velocity, 512, 512])
		batch_coarse = down_sampler(batch).reshape([b_size, 
													num_time, 
													num_velocity,
													args.coarse_dim[0], 
													args.coarse_dim[1]])
		batch_coarse_flatten = batch_coarse.reshape([b_size, 
													 num_time,
													 num_velocity * args.coarse_dim[0] * args.coarse_dim[1]])
		assert num_time == args.n_ctx + 1
		for j in (range(num_time - args.n_ctx)):
			model.train()
			optimizer.zero_grad()
			xn = batch_coarse_flatten[:,j:j+args.n_ctx,:]
			xnp1,_,_,_=model(inputs_embeds = xn, past=None)
			xn_label = batch_coarse_flatten[:,j+1:j+1+args.n_ctx,:]
			loss = loss_func(xnp1, xn_label)
			loss.backward()
			optimizer.step()
	return model