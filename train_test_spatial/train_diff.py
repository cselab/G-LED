from tqdm import tqdm
import torch
import pdb
import sys
import os
import numpy as np
sys.path.insert(0, './util')
from utils import save_loss

def train_diff(diff_args,
			   seq_args,
			   trainer,
			   data_loader):
	loss_list = []
	for epoch in range(diff_args.epoch_num):
		down_sampler = torch.nn.Upsample(size=seq_args.coarse_dim, 
								     	 mode=seq_args.coarse_mode)
		up_sampler   = torch.nn.Upsample(size=[512, 512], 
								     	 mode=seq_args.coarse_mode)
		model, loss = train_epoch(diff_args,seq_args, trainer, data_loader,down_sampler,up_sampler)
		if epoch % 1 ==0 and epoch > 0:
			save_loss(diff_args, loss_list+[loss],epoch)
			model.save(path=os.path.join(diff_args.model_save_path, 
										 'model_epoch_' + str(epoch)))
		if epoch >= 1:
			if loss < min(loss_list):
				save_loss(diff_args, loss_list+[loss],epoch)
				model.save(path=os.path.join(diff_args.model_save_path, 
											'best_model_sofar'))
				np.savetxt(os.path.join(diff_args.model_save_path, 
									'best_model_sofar_epoch'),np.ones(2)*epoch)
		loss_list.append(loss) 
		print("finish training epoch {}".format(epoch))


def train_epoch(diff_args,seq_args, trainer, data_loader,down_sampler,up_sampler):
	loss_epoch = []
	print('Iteration is ', len(data_loader))
	for iteration, batch in tqdm(enumerate(data_loader)):
		batch = batch.to(diff_args.device).float()
		bsize = batch.shape[0]
		ntime = batch.shape[1] 
		batch_coarse      = down_sampler(batch.reshape([bsize*ntime,2,512,512]))
		batch_coarse2fine = up_sampler(batch_coarse).reshape(batch.shape)
		#need # B x F x T x H x W
		batch= batch.permute([0,2,1,3,4])
		batch_coarse2fine = batch_coarse2fine.permute([0,2,1,3,4])
		#print(batch.device)
		loss=trainer(batch,cond_images=batch_coarse2fine,unet_number=1,ignore_time=False)
		trainer.update(unet_number=1)
		loss_epoch.append(loss)
	return trainer, sum(loss_epoch)/len(loss_epoch)
