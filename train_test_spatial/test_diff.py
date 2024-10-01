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

def test_final_overall(args_final, 
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
	test_final(args_final, 
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

def test_final(args_final, 
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
				pdb.set_trace()
				
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
				pickle.dump(DIC, open(contour_dir+'/'+seq_name+"/DIC.npy", 'wb'), protocol=4)
				
				for d in tqdm(range(num_velocity+1)):
					try:
						os.makedirs(contour_dir+'/'+seq_name+'/'+str(d))
					except:
						pass
					sub_seq_name = contour_dir+'/'+seq_name+'/'+str(d)
					for t in tqdm(range(Nt)):
						#pdb.set_trace()
						if d == 2:
							velo_micro_truth = np.sqrt(truth_micro[0,0,t,:,:].cpu().numpy()**2+truth_micro[0,1,t,:,:].cpu().numpy()**2)
							velo_micro_led   = np.sqrt(prediction_micro[0,0,t,:,:].cpu().numpy()**2+prediction_micro[0,1,t,:,:].cpu().numpy()**2)
							velo_micro_rcons = np.sqrt(recon_micro[0,0,t,:,:].cpu().numpy()**2+recon_micro[0,1,t,:,:].cpu().numpy()**2)
							velo_macro_truth = np.sqrt(truth[0,t,0,:,:].cpu().numpy()**2+truth[0,t,1,:,:].cpu().numpy()**2)
							velo_macro_led   = np.sqrt(prediction[0,t,0,:,:].cpu().numpy()**2+prediction[0,t,1,:,:].cpu().numpy()**2)
							vmin =0# truth[0,:,d,:,:].cpu().numpy().min()
							vmax =20# truth[0,:,d,:,:].cpu().numpy().max()
						else:
							velo_micro_truth = truth_micro[0,d,t,:,:].cpu().numpy()
							velo_micro_led   = prediction_micro[0,d,t,:,:].cpu().numpy()
							velo_micro_rcons = recon_micro[0,d,t,:,:].cpu().numpy()
							velo_macro_truth = truth[0,t,d,:,:].cpu().numpy()
							velo_macro_led   = prediction[0,t,d,:,:].cpu().numpy()
							vmin = truth[0,:,d,:,:].cpu().numpy().min()
							vmax = truth[0,:,d,:,:].cpu().numpy().max()

						fig, axes = plt.subplots(nrows=5, ncols=1)
						fig.subplots_adjust(hspace=0.6)
						norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
						im0 = axes[0].imshow(velo_macro_led[:,:],extent=[0,10,0,2],cmap = 'jet',interpolation=None,norm=norm)
						axes[0].set_title('LED Macro')
						axes[0].set_ylabel('y')
						axes[0].set_xticks([])
						
						im1 = axes[1].imshow(velo_macro_truth[:,:],extent=[0,10,0,2], cmap = 'jet',interpolation=None,norm=norm)
						axes[1].set_title('Truth Macro')#,rotation=-90, position=(1, -1), ha='left', va='center')
						axes[1].set_ylabel('y')
						axes[1].set_xticks([])

						
						im2 = axes[2].imshow(velo_micro_led[:,:],extent=[0,10,0,2], cmap = 'jet',interpolation='bicubic',norm=norm)
						axes[2].set_title('LED Micro')#,rotation=-90, position=(1, -1), ha='left', va='center')
						axes[2].set_ylabel('y')
						axes[2].set_xticks([])

						im3 = axes[3].imshow(velo_micro_rcons[:,:],extent=[0,10,0,2], cmap = 'jet',interpolation='bicubic',norm=norm)
						axes[3].set_title('Reconstruction Micro')#,rotation=-90, position=(1, -1), ha='left', va='center')
						axes[3].set_ylabel('y')
						axes[3].set_xticks([])

						im4 = axes[4].imshow(velo_micro_truth,extent=[0,10,0,2], cmap = 'jet',interpolation='bicubic',norm=norm)
						axes[4].set_title('Truth Micro')#,rotation=-90, position=(1, -1), ha='left', va='center')
						axes[4].set_xlabel('x')
						axes[4].set_ylabel('y')
						
						fig.subplots_adjust(right=0.8)
						fig.colorbar(im0,orientation="vertical",ax = axes)
						#fig.set_size_inches(20, 15, forward=True)
						fig.savefig(sub_seq_name+'/time'+str(t)+'.png', bbox_inches='tight',dpi=500)
						plt.close(fig)









				# data is nn prediction
			# 	data = mem[i].cpu().numpy()
			# 	X_AR = batch_coarse_merge_spatial[i,previous_len:previous_len+Nt+1,:].cpu().numpy()
			# 	data = data.reshape([-1,16])
			# 	X_AR = X_AR.reshape([-1,16])	
			# 	pdb.set_trace()

			# 	mem  = mem.reshape([mem.shape[0],
			# 									1,
			# 									mem.shape[1],
			# 									mem.shape[2]])
			# mem2fine = []
			# for nc in range(int(mem.shape[2]/args_diff.Nt)):
			# 	mem2fine.append(up_sampler(mem[:,:,nc*args_diff.Nt:(nc+1)*args_diff.Nt])) 
			# mem2fine = torch.cat(mem2fine,dim=2)
			
			# mem2fine = mem2fine.reshape([mem2fine.shape[0],
			# 								1,
			# 								1,
			# 								mem2fine.shape[-2],
			# 								mem2fine.shape[-1]])




"""
def test_final(args_final, 
			   args_seq, 
			   args_diff, 
			   trainer, 
			   model, 
			   data_loader,
			   down_sampler,up_sampler):
	print('Iteration is ', len(data_loader))
	IDHistory = [i for i in range(1, args_seq.n_ctx)]
	with torch.no_grad():
		for iteration, batch in tqdm(enumerate(data_loader)):	
			batch_coarse = down_sampler(batch)
			bcfs = [batch_coarse.shape[0],batch_coarse.shape[1],args_seq.coarse_dim[0]*args_seq.coarse_dim[1]]
			batch_coarse_flatten = batch_coarse.reshape(bcfs)

			


			
			bffs = [batch.shape[0], batch.shape[1], 64]
			batch_fine_flatten   = batch.reshape(bffs)

			len_batch = batch_fine_flatten.shape[1]

			coarse_one  = batch_coarse_flatten[:,:args_final.test_Nt,:]

			if args_final.warm_up:
				_,past,_,_=model(inputs_embeds = coarse_one[:,0:args_seq.n_ctx-1,:], past=None)
				xn = coarse_one[:,args_seq.n_ctx:args_seq.n_ctx+1,:]
				previous_len = args_seq.n_ctx
			else:
				past = None
				xn = coarse_one[:,0:1,:]
				previous_len = 1 
			mem = []
			for j in tqdm(range(args_final.test_Nt-1)):
				if j == 0:
					xnp1,past,_,_=model(inputs_embeds = xn, past=past)
				elif past[0][0].shape[2] < args_seq.n_ctx and j > 0:
					if args_final.warm_up:
						raise ValueError("Should not stop here")
					xnp1,past,_,_=model(inputs_embeds = xn, past=past)
				else:
					past = [[past[l][0][:,:,IDHistory,:], past[l][1][:,:,IDHistory,:]] for l in range(args_seq.n_layer)]
					xnp1,past,_,_=model(inputs_embeds = xn, past=past)
				xn = xnp1
				mem.append(xn)
			mem=torch.cat([coarse_one[:,0:1,:]]+mem,dim=1)

			


			mem  = mem.reshape([mem.shape[0],
												1,
												mem.shape[1],
												mem.shape[2]])
			mem2fine = []
			for nc in range(int(mem.shape[2]/args_diff.Nt)):
				mem2fine.append(up_sampler(mem[:,:,nc*args_diff.Nt:(nc+1)*args_diff.Nt])) 
			mem2fine = torch.cat(mem2fine,dim=2)
			
			mem2fine = mem2fine.reshape([mem2fine.shape[0],
											1,
											1,
											mem2fine.shape[-2],
											mem2fine.shape[-1]])

			coarse_one  = coarse_one.reshape([coarse_one.shape[0],
												1,
												coarse_one.shape[1],
												coarse_one.shape[2]])
			
			coarse2fine = []

			for nc in range(int(coarse_one.shape[2]/args_diff.Nt)):
				coarse2fine.append(up_sampler(coarse_one[:,:,nc*args_diff.Nt:(nc+1)*args_diff.Nt])) 
			coarse2fine = torch.cat(coarse2fine,dim=2)
			
			
			fine_one    = batch_fine_flatten[:,:args_final.test_Nt]
			fine_one    = fine_one.reshape([fine_one.shape[0],
											1,
											1,
											fine_one.shape[1],
											fine_one.shape[2]])
			
			coarse2fine = coarse2fine.reshape([fine_one.shape[0],
											1,
											1,
											fine_one.shape[-2],
											fine_one.shape[-1]])
			data = []
			data_led = []
			for nc in tqdm(range(int(coarse_one.shape[2]/args_diff.Nt))):
				# B x T x F x H x W
				les_video_sampled_chunck = trainer.sample(video_frames=1, 
				                                          cond_images=coarse2fine[:,:,:,nc*args_diff.Nt:(nc+1)*args_diff.Nt])
				les_video_sampled_chunck_led = trainer.sample(video_frames=1, 
				                                          cond_images=mem2fine[:,:,:,nc*args_diff.Nt:(nc+1)*args_diff.Nt])
				data.append(les_video_sampled_chunck[0,0,0].cpu().numpy())
				data_led.append(les_video_sampled_chunck_led[0,0,0].cpu().numpy())
			data = np.vstack(data)
			data_led = np.vstack(data_led)
			X_AR = fine_one[0,0,0].cpu().numpy()
			coarse2fine_np = coarse2fine[0,0,0].cpu().numpy()
			mem2fine_np    = mem2fine[0,0,0].detach().cpu().numpy()	

			asp = data.shape[1]/data.shape[0]*3
			fig, axes = plt.subplots(nrows=1, ncols=5)
			norm = matplotlib.colors.Normalize(vmin=X_AR.min(), vmax=X_AR.max())
			im0 = axes[0].imshow(data[:,:],
								aspect=asp,cmap = 'jet',interpolation='bicubic',norm=norm,extent=[0, 16, data.shape[0]*0.25, 0])
			axes[0].title.set_text('micro \n reconstruction')
			axes[0].invert_yaxis()
			axes[0].set_xlabel('x')
			axes[0].set_ylabel('t')
			axes[0].set_xticks([])
			axes[0].set_yticks([])

			im1 = axes[1].imshow(data_led[:,:],
								aspect=asp,cmap = 'jet',interpolation='bicubic',norm=norm,extent=[0, 16, data.shape[0]*0.25, 0])
			axes[1].title.set_text('micro \n LED')
			axes[1].invert_yaxis()
			axes[1].set_xlabel('x')
			axes[1].set_xticks([])
			axes[1].set_yticks([])
			
			
			im2 = axes[2].imshow(X_AR[:,:],aspect=asp, cmap = 'jet',interpolation='bicubic',norm=norm,extent=[0, 16, data.shape[0]*0.25, 0])
			axes[2].title.set_text('micro \n truth')
			axes[2].invert_yaxis()
			axes[2].set_xlabel('x')
			axes[2].set_yticks([])
			axes[2].set_xticks([])
			
			# im2 = axes[3].imshow(np.abs(X_AR-data),aspect=asp, cmap = 'jet',interpolation='bicubic',norm=norm,extent=[0, 16, data.shape[0]*0.25, 0])
			# axes[3].title.set_text('En/De- \n Coder \n Error')
			# axes[3].invert_yaxis()
			# axes[3].set_xlabel('x')
			# axes[3].set_yticks([])

			im3 = axes[3].imshow(coarse2fine_np,aspect=asp, cmap = 'jet',interpolation='bicubic',norm=norm,extent=[0, 16, data.shape[0]*0.25, 0])
			axes[3].title.set_text('macro \n truth')
			axes[3].invert_yaxis()
			axes[3].set_xlabel('x')
			axes[3].set_yticks([])
			axes[3].set_xticks([])

			im4 = axes[4].imshow(mem2fine_np,aspect=asp, cmap = 'jet',interpolation='bicubic',norm=norm,extent=[0, 16, data.shape[0]*0.25, 0])
			axes[4].title.set_text('macro \n LED')
			axes[4].invert_yaxis()
			axes[4].set_xlabel('x')
			axes[4].set_yticks([])
			axes[4].set_xticks([])
			
			fig.subplots_adjust(right=0.8)
			fig.colorbar(im0,orientation="horizontal",ax = axes)
			fig.savefig('./batch'+str(iteration)+'sample_kas'+'Nt_read'+str(args_final.Nt_read)+'.png', bbox_inches='tight',dpi=500)
			plt.close(fig)


			fig, axes = plt.subplots(nrows=1, ncols=4)
			norm = matplotlib.colors.Normalize(vmin=X_AR.min(), vmax=X_AR.max())
			im0 = axes[0].imshow(data[:,:],
								aspect=asp,cmap = 'jet',interpolation='bicubic',norm=norm,extent=[0, 16, data.shape[0]*0.25, 0])
			axes[0].title.set_text('decoder \n reconstruction')
			axes[0].invert_yaxis()
			axes[0].set_xlabel('x')
			axes[0].set_ylabel('t')
			axes[0].set_xticks([])
			axes[0].set_yticks([])

			
			
			im2 = axes[2].imshow(X_AR[:,:],aspect=asp, cmap = 'jet',interpolation='bicubic',norm=norm,extent=[0, 16, data.shape[0]*0.25, 0])
			axes[2].title.set_text('micro \n truth')
			axes[2].invert_yaxis()
			axes[2].set_xlabel('x')
			axes[2].set_yticks([])
			axes[2].set_xticks([])
			
			# im2 = axes[3].imshow(np.abs(X_AR-data),aspect=asp, cmap = 'jet',interpolation='bicubic',norm=norm,extent=[0, 16, data.shape[0]*0.25, 0])
			# axes[3].title.set_text('En/De- \n Coder \n Error')
			# axes[3].invert_yaxis()
			# axes[3].set_xlabel('x')
			# axes[3].set_yticks([])

			im3 = axes[1].imshow(coarse2fine_np,aspect=asp, cmap = 'jet',interpolation='bicubic',norm=norm,extent=[0, 16, data.shape[0]*0.25, 0])
			axes[1].title.set_text('encoder \n output')
			axes[1].invert_yaxis()
			axes[1].set_xlabel('x')
			axes[1].set_yticks([])
			axes[1].set_xticks([])

			im4 = axes[3].imshow(np.abs(data[:,:]-X_AR[:,:]),aspect=asp, cmap = 'jet',interpolation='bicubic',norm=norm,extent=[0, 16, data.shape[0]*0.25, 0])
			axes[3].title.set_text('En/De-coder \n error')
			axes[3].invert_yaxis()
			axes[3].set_xlabel('x')
			axes[3].set_yticks([])
			axes[3].set_xticks([])
			
			fig.subplots_adjust(right=0.8)
			fig.colorbar(im0,orientation="horizontal",ax = axes)
			fig.savefig('./batch'+str(iteration)+'sample_kas'+'Nt_read'+str(args_final.Nt_read)+'endecoder.png', bbox_inches='tight',dpi=500)
			plt.close(fig)
			pdb.set_trace()

"""			