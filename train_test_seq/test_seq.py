import pdb
import torch
from tqdm import tqdm
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import matplotlib
"""
Start training test
"""
def test_epoch(args,
			   model, 
			   data_loader,
			   loss_func,
			   Nt,
			   down_sampler,
			   ite_thold = None):
	with torch.no_grad():
		#IDHistory = [0] + [i for i in range(2, args.n_ctx)]
		IDHistory = [i for i in range(1, args.n_ctx)]
		REs = []
		print("Total ite", len(data_loader))
		for iteration, batch in tqdm(enumerate(data_loader)):
			if ite_thold is None:
				pass
			else:
				if iteration>ite_thold:
					break
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
			
			past = None
			xn = batch_coarse_flatten[:,0:1,:]
			previous_len = 1 
			mem = []
			for j in (range(Nt)):
				if j == 0:
					xnp1,past,_,_=model(inputs_embeds = xn, past=past)
				elif past[0][0].shape[2] < args.n_ctx and j > 0:
					xnp1,past,_,_=model(inputs_embeds = xn, past=past)
				else:
					past = [[past[l][0][:,:,IDHistory,:], past[l][1][:,:,IDHistory,:]] for l in range(args.n_layer)]
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
	return max(REs),min(REs),sum(REs)/len(REs),3*np.std(np.asarray(REs))







def test_plot_eval(args,
				   args_sample,
				   model, 
				   data_loader,
				   loss_func,
				   Nt,
				   down_sampler):
	try:
		os.makedirs(args.experiment_path+'/contour')
	except:
		pass
	contour_dir = args.experiment_path+'/contour'
	with torch.no_grad():
		#IDHistory = [0] + [i for i in range(2, args.n_ctx)]
		IDHistory = [i for i in range(1, args.n_ctx)]
		REs = []
		print("Total ite", len(data_loader))
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
			
			past = None
			xn = batch_coarse_flatten[:,0:1,:]
			previous_len = 1 
			mem = []
			for j in (range(Nt)):
				if j == 0:
					xnp1,past,_,_=model(inputs_embeds = xn, past=past)
				elif past[0][0].shape[2] < args.n_ctx and j > 0:
					xnp1,past,_,_=model(inputs_embeds = xn, past=past)
				else:
					past = [[past[l][0][:,:,IDHistory,:], past[l][1][:,:,IDHistory,:]] for l in range(args.n_layer)]
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
												 args.coarse_dim[0],
												 args.coarse_dim[1]])
				truth = truth.reshape([truth.shape[0],
									   truth.shape[1],
									   num_velocity,
									   args.coarse_dim[0],
									   args.coarse_dim[1]])
				
				seq_name = 'batch'+str(iteration)+'sample'+str(i)
				try:
					os.makedirs(contour_dir+'/'+seq_name)
				except:
					pass
				for d in tqdm(range(num_velocity+1)):
					try:
						os.makedirs(contour_dir+'/'+seq_name+'/'+str(d))
					except:
						pass
					sub_seq_name = contour_dir+'/'+seq_name+'/'+str(d)
					for t in tqdm(range(Nt)):
						
						
						if d == 2:
							data = np.sqrt(prediction[0,t,0,:,:].cpu().numpy()**2+prediction[0,t,1,:,:].cpu().numpy()**2)
						else:
							data = prediction[0,t,d,:,:].cpu().numpy()
						
						if d == 2:
							X_AR = np.sqrt(truth[0,t,0,:,:].cpu().numpy()**2+truth[0,t,1,:,:].cpu().numpy()**2)
						else:
							X_AR = truth[0,t,d,:,:].cpu().numpy()
						#asp = data.shape[1]/data.shape[0]
						fig, axes = plt.subplots(nrows=3, ncols=1)
						fig.subplots_adjust(hspace=0.5)
						norm = matplotlib.colors.Normalize(vmin=X_AR.min(), vmax=X_AR.max())
						im0 = axes[0].imshow(data[:,:],extent=[0,10,0,2],cmap = 'jet',interpolation='bicubic',norm=norm)
						axes[0].set_title('LED Macro')#,rotation=-90, position=(1, -1))#, ha='left', va='center')
						#axes[0].invert_yaxis()
						#axes[0].set_xlabel('x')
						axes[0].set_ylabel('y')
						axes[0].set_xticks([])
						
						im1 = axes[1].imshow(X_AR[:,:],extent=[0,10,0,2], cmap = 'jet',interpolation='bicubic',norm=norm)
						axes[1].set_title('Label')#,rotation=-90, position=(1, -1), ha='left', va='center')
						#axes[1].invert_yaxis()
						axes[1].set_ylabel('y')
						axes[1].set_xticks([])

						im2 = axes[2].imshow(np.abs(X_AR-data),extent=[0,10,0,2], cmap = 'jet',interpolation='bicubic',norm=norm)
						axes[2].set_title('Error')#,rotation=-90, position=(1, -1), ha='left', va='center')
						#axes[2].invert_yaxis()
						axes[2].set_xlabel('x')
						axes[2].set_ylabel('y')
						#axes[2].set_xticks([])
						
						fig.subplots_adjust(right=0.8)
						fig.colorbar(im0,orientation="horizontal",ax = axes)
						fig.savefig(sub_seq_name+'/time'+str(t)+'.png', bbox_inches='tight',dpi=500)
						plt.close(fig)

"""
start test
"""
def eval_seq_overall(args_train,
					 args_sample,
					 model, 
					 data_loader, 
					 loss_func):
	down_sampler = torch.nn.Upsample(size=args_train.coarse_dim, 
								     mode=args_train.coarse_mode)
	Nt = args_sample.test_Nt
	tic = time.time()
	print('Start test forwarding with Step number of ', Nt)
	max_mre,min_mre, mean_mre, sigma3 = test_epoch(args=args_train,
												   model=model, 
												   data_loader=data_loader,
												   loss_func=loss_func,
												   Nt=Nt,
												   down_sampler=down_sampler,
												   ite_thold = None)
	print('#### max mre test####=',max_mre)
	print('#### mean mre test####=',mean_mre)
	print('#### min mre test####=',min_mre)
	print('#### 3 sigma ####=',sigma3)
	print('Test elapsed ', time.time()-tic)
	test_plot_eval(args=args_train,
				   args_sample = args_sample,
				   model=model, 
				   data_loader=data_loader,
				   loss_func=loss_func,
				   Nt=Nt,
				   down_sampler=down_sampler)