from tqdm import tqdm
import torch
import pdb
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import os

sys.path.insert(0, './util')
from utils import save_loss

def test_final_overall_ensamble(args_final, 
								args_seq, 
								args_diff, 
								trainer, 
								seq_model,
								data_loader):
	down_sampler = torch.nn.Upsample(size=args_seq.coarse_dim, 
									 mode=args_seq.coarse_mode)
	up_sampler   = torch.nn.Upsample(size=[512, 512], 
									 mode=args_seq.coarse_mode)
	#max_mre,min_mre, mean_mre, sigma3 = 
	test_final_ensamble(args_final, 
			   args_seq, 
			   args_diff, 
			   trainer, 
			   seq_model, 
			   data_loader,
			   down_sampler,
			   up_sampler)
	#print('#### max mre####=',max_mre)
	#print('#### mean mre####=',mean_mre)
	#print('#### min mre####=',min_mre)
	#print('#### 3 sigma ####=',sigma3)
	pdb.set_trace()

def test_final_ensamble(args_final, 
			   args_seq, 
			   args_diff, 
			   trainer, 
			   model, 
			   data_loader,
			   down_sampler,
			   up_sampler,
			   save_flag=True):
	try:
		os.makedirs(args_final.experiment_path+'/contour')
	except:
		pass
	contour_dir = args_final.experiment_path+'/contour'
	loss_func = torch.nn.MSELoss()
	Nt = args_final.test_Nt
	for I in range(10):
		LED_micro_list = []
		LED_macro_list = []
		Dif_recon_list = []
		CFD_micro_list = []
		CFD_macro_list = []
		with torch.no_grad():
			IDHistory = [i for i in range(1, args_seq.n_ctx)]
			REs = []
			REs_fine = []
			print('total ite', len(data_loader))
			for iteration, batch in tqdm(enumerate(data_loader)):
				batch = batch.to(args_final.device).float()
				b_size = batch.shape[0]
				assert b_size == 1
				num_time = batch.shape[1]
				num_velocity = 2
				batch = batch.reshape([b_size*num_time, num_velocity, 512, 512])


				batch_coarse = down_sampler(batch).reshape([b_size, 
															num_time, 
															num_velocity,
															args_seq.coarse_dim[0], 
															args_seq.coarse_dim[1]])
				
				
				batch_coarse_flatten = batch_coarse.reshape([b_size, 
															num_time,
															num_velocity * args_seq.coarse_dim[0] * args_seq.coarse_dim[1]])
				
				past = None
				xn = batch_coarse_flatten[:,0:1,:]
				previous_len = 1 
				mem = []
				for j in tqdm(range(Nt)):
					if j == 0:
						xnp1,past,_,_=model(inputs_embeds = xn, past=past)
					elif past[0][0].shape[2] < args_seq.n_ctx and j > 0:
						xnp1,past,_,_=model(inputs_embeds = xn, past=past)
					else:
						past = [[past[l][0][:,:,IDHistory,:], past[l][1][:,:,IDHistory,:]] for l in range(args_seq.n_layer)]
						xnp1,past,_,_=model(inputs_embeds = xn, past=past)
					xn = xnp1
					mem.append(xn)
				mem=torch.cat(mem,dim=1)
				local_batch_size = mem.shape[0]
				for i in tqdm(range(local_batch_size)):
					er = loss_func(mem[i:i+1],
								batch_coarse_flatten[i:i+1,previous_len:previous_len+Nt,:])
					r_er = er/loss_func(mem[i:i+1]*0,
										batch_coarse_flatten[i:i+1,previous_len:previous_len+Nt,:])
					REs.append(r_er.item())
					prediction = mem[i:i+1]
					truth      = batch_coarse_flatten[i:i+1,previous_len:previous_len+Nt,:]
					# spatial recover
					prediction = prediction.reshape([prediction.shape[0],
													prediction.shape[1],
													num_velocity,
													args_seq.coarse_dim[0],
													args_seq.coarse_dim[1]])
					truth = truth.reshape([truth.shape[0],
										truth.shape[1],
										num_velocity,
										args_seq.coarse_dim[0],
										args_seq.coarse_dim[1]])

					

					assert prediction.shape[0] == truth.shape[0] == 1
					bsize_here = 1
					ntime      = prediction.shape[1]

					prediction_macro = up_sampler(prediction[0]).reshape([bsize_here, ntime, num_velocity, 512, 512])
					truth_macro      = up_sampler(truth[0]).reshape([bsize_here, ntime, num_velocity, 512, 512])



					prediction_macro = prediction_macro.permute([0,2,1,3,4])
					truth_macro      = truth_macro.permute([0,2,1,3,4])
					truth_micro      = batch.permute([1,0,2,3]).unsqueeze(0)[:,:,1:]
					
					recon_micro = []
					prediction_micro = []
					#pdb.set_trace()
					vf = args_diff.Nt
					Nvf = args_final.test_Nt//vf
					for j in range(Nvf):
						recon_micro.append(trainer.sample(video_frames=vf, cond_images=truth_macro[:,:,vf*j:vf*j+vf]))
						prediction_micro.append(trainer.sample(video_frames=vf, cond_images=prediction_macro[:,:,vf*j:vf*j+vf]))
					recon_micro = torch.cat(recon_micro,dim=2)
					prediction_micro = torch.cat(prediction_micro,dim=2)
					#pdb.set_trace()
					
					seq_name = 'batch'+str(iteration)+'sample'+str(i)
					try:
						os.makedirs(contour_dir+'/'+seq_name)
					except:
						pass

					DIC = {"prediction_micro":prediction_micro.detach().cpu().numpy(),
						"recon_micro":recon_micro.detach().cpu().numpy(),
						"truth_micro":truth_micro.detach().cpu().numpy(),
						"truth_macro":truth_macro.detach().cpu().numpy(),
						"prediction_macro":prediction_macro.detach().cpu().numpy()}
					pickle.dump(DIC, open(contour_dir+'/'+seq_name+"/DIC"+str(I)+".npy", 'wb'), protocol=4)
				
				







