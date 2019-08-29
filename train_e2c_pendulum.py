from e2c import datasets,e2c,utils
import argparse
from torch.utils.data import DataLoader
from torch import optim
import torch
import os
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import numpy as np
import json
import visualize_e2c_pendulum
import cv2

parser = argparse.ArgumentParser(description='train vae')

log = parser.add_argument_group('logger')
train = parser.add_argument_group('train')
model = parser.add_argument_group('model')

parser.add_argument(
	'--gpu-ids',
	type=int,
	default=[0,1,2,3],
	nargs='+',
	help='GPUs to use [-1 CPU only] (default: -1)')

train.add_argument('--epoch',type=int,default=500,metavar='TS',help='training steps')
train.add_argument('--bs',type=int,default=256,metavar='BS',help='training batch size')
train.add_argument('--lr',type=float,default=1e-3,metavar='lr',help='learning rate')
train.add_argument('--loss',default=e2c.compute_loss,metavar='LOSS',help='loss function')

log.add_argument('--log-dir',type=str,default='/home/pikey/Data/e2c/e2clog',help='log directory')

model.add_argument('--z-dim',type=int,default=2,metavar='dz',help='latent_space_dimension')

def get_args():
	return parser.parse_args()


if __name__ == '__main__':
	torch.manual_seed(1)
	args = get_args()

	#save args
	config = vars(args)
	if not os.path.exists(args.log_dir):
		os.makedirs(args.log_dir)
	with open(os.path.join(args.log_dir,'config.json'),'wt') as f:
		json.dump(config,f,cls=utils.DataEnc,indent=2)

	gpu=0
	writer = SummaryWriter(log_dir=os.path.join(args.log_dir,'train_result'))

	#todo data preparation
	dataset = datasets.GymPendulumDatasetV2(dir='/home/pikey/Data/e2c/dataset/pendulum')
	dataloader = DataLoader(dataset,args.bs,shuffle=True,
                        num_workers=16,drop_last=True,pin_memory=True)

	test_dataset = datasets.GymPendulumDatasetV2(dir='/home/pikey/Data/e2c/visual_dataset_pendulum')
	test_dataloader = DataLoader(test_dataset, args.bs, shuffle=True,
	                        num_workers=16, drop_last=False,pin_memory=True)
	model = e2c.E2C(datasets.GymPendulumDatasetV2.height*datasets.GymPendulumDatasetV2.width,dim_z=args.z_dim,dim_u=1)
	model.to(gpu)
	opt = optim.Adam(model.parameters(),lr=args.lr)

	step= 0
	for i in range(args.epoch):
		j = 0
		for before,a,r,after in dataloader:
			a = a.float()
			step+=1
			j+=1
			next_pre_rec = model(before.view(args.bs,-1).to(gpu),a.to(gpu),after.view(args.bs,-1).to(gpu))
			loss_rec,loss_trans  =e2c.compute_loss(model.x_dec,model.x_next_pred_dec,model.x_next_dec,
			                                    before.view(args.bs,-1).to(gpu),after.view(args.bs,-1).to(gpu),
			                                    model.Qz,model.Qz_next_pred,model.Qz_next,mse=True)
			loss = loss_rec+loss_trans
			opt.zero_grad()
			loss.backward()
			opt.step()
			print('loss_rec:{}, loss_trans:{}'.format(loss_rec.item(),loss_trans.item()))

			#todo log
			#save img
			if step%10==0:
				img = next_pre_rec[0].detach().cpu().view(1, datasets.GymPendulumDatasetV2.height,
				                                          datasets.GymPendulumDatasetV2.width) \
					.numpy()
				img = np.append(after[0].numpy(),img,axis=1)
				# cv2.imshow('rec',img.transpose(1,2,0))
				# cv2.imshow('raw',after[0].numpy().transpose(1,2,0))
				# cv2.waitKey(1)
				rec_file = os.path.join(args.log_dir, 'img', 'epoch{:03}_batch{:05d}.jpg'.format(i, j))
				if not os.path.exists(os.path.join(args.log_dir, 'img')):
					os.makedirs(os.path.join(args.log_dir, 'img'))
				plt.imsave(rec_file, img.copy().squeeze())

				writer.add_image('raw_rec',img,step)
			#weight histogram
			# for _, (name,param) in enumerate(model.named_parameters()):
			# 	if 'bn' not in name:
			# 		writer.add_histogram(name,param,step)
			#loss
			writer.add_scalar('loss',loss.item(),step)
			writer.add_scalar('loss_rec',loss_rec.item(),step)
			writer.add_scalar('loss_trans',loss_trans.item(),step)
			if step%1000==0:
				#todo save the test results
				fig_z = visualize_e2c_pendulum.visualize(model, test_dataloader,gpu=gpu)
				utils.save_fig(fig_z,os.path.join(args.log_dir,'fig_z'),str(step)+'.png')

		if not os.path.exists(os.path.join(args.log_dir, 'model')):
			os.makedirs(os.path.join(args.log_dir, 'model'))
		torch.save(model.state_dict(),os.path.join(args.log_dir,'model','epoch{}.pkl'.format(i)))
	writer.close()

