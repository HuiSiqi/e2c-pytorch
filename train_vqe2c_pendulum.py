from e2c import datasets,vqe2c,utils
import argparse
from torch.utils.data import DataLoader
from torch import optim
import torch
import os
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import numpy as np
import json
import cv2
import visualize_vqe2c_pendulum

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
train.add_argument('--loss',default=vqe2c.compute_loss,metavar='LOSS',help='loss function')

log.add_argument('--log-dir',type=str,default='/home/pikey/Data/e2c/vqe2c-log',help='log directory')

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

	test_dataset = datasets.GymPendulumDatasetV2_visual('/home/pikey/Data/e2c/visual_dataset_pendulum')
	test_dataloader = DataLoader(test_dataset,args.bs,shuffle=False,num_workers=256,drop_last=False,pin_memory=True)

	model = vqe2c.VQE2C(datasets.GymPendulumDatasetV2.height*datasets.GymPendulumDatasetV2.width,
	                    dim_z=args.z_dim,dim_u=1,topic_num=4)
	model.to(gpu).train()
	opt = optim.Adam(model.parameters(),lr=args.lr)

	step= 0
	for i in range(args.epoch):
		j = 0
		for before,a,r,after in dataloader:
			a = a.float()
			step+=1
			j+=1
			next_pre_rec = model(before.view(args.bs,-1).to(gpu),a.to(gpu),after.view(args.bs,-1).to(gpu))
			loss_rec,loss_trans,loss_commit =vqe2c.compute_loss(model.x_dec,model.x_next_pred_dec,model.x_next_dec,
			                                    before.view(args.bs,-1).to(gpu),after.view(args.bs,-1).to(gpu),
			                                    model.Qz,model.Qz_next_pred,model.Qz_next,model.z_diff,model.z_next_diff,mse=True)
			loss = loss_rec+loss_trans+loss_commit
			opt.zero_grad()
			loss.backward()
			opt.step()
			print('loss_rec:{}, loss_trans:{}, loss_commit:{}'.format(loss_rec.item(),loss_trans.item(),loss_commit.item()))
			# print('embed:{}'.format(model.quantize.embed.cpu().numpy()))
			#todo log
			#save img
			if step%10==0:
				#save_prediction_contrast
				img = next_pre_rec[0].detach().cpu().view(1, datasets.GymPendulumDatasetV2.height,
				                                          datasets.GymPendulumDatasetV2.width) \
					.numpy()
				img = np.append(after[0].numpy(),img,axis=1)
				# cv2.imshow('rec',img.transpose(1,2,0))
				# cv2.imshow('raw',after[0].numpy().transpose(1,2,0))
				# cv2.waitKey(1)
				dir = os.path.join(args.log_dir,'img_dec_z_next_pre')
				rec_file = 'epoch{:03}_batch{:05d}.jpg'.format(i, j)
				utils.save_img(img,dir,rec_file)
				writer.add_image('dec_z_next_pre', img, step)

				#save x and dec_z
				img = model.x_dec[0].detach().cpu().view(1, datasets.GymPendulumDatasetV2.height,
				                                          datasets.GymPendulumDatasetV2.width) \
					.numpy()
				img = np.append(before[0].numpy(), img, axis=1)
				dir = os.path.join(args.log_dir, 'img_dec_z')
				utils.save_img(img, dir, rec_file)
				writer.add_image('dec_z', img, step)

				#save x_next and dec_z_next
				img = model.x_next_dec[0].detach().cpu().view(1, datasets.GymPendulumDatasetV2.height,
				                                         datasets.GymPendulumDatasetV2.width) \
					.numpy()
				img = np.append(after[0].numpy(), img, axis=1)
				dir = os.path.join(args.log_dir, 'img_dec_z_next')
				utils.save_img(img, dir, rec_file)
				writer.add_image('dec_z_next',img,step)

			if step%1000==0:
				#todo save the test results
				fig_x_cls,fig_z,fig_z_cls,fig_t_c = visualize_vqe2c_pendulum.visualize(model, test_dataloader,gpu=gpu)
				utils.save_fig(fig_x_cls,os.path.join(args.log_dir,'fig_x_cls'),str(step)+'.png')
				utils.save_fig(fig_z_cls,os.path.join(args.log_dir,'fig_z_cls'),str(step)+'.png')
				utils.save_fig(fig_z,os.path.join(args.log_dir,'fig_z'),str(step)+'.png')
				utils.save_fig(fig_t_c,os.path.join(args.log_dir,'fig_t_c'),str(step)+'.png')

			#weight histogram
			# for _, (name,param) in enumerate(model.named_parameters()):
			# 	if 'bn' not in name:
			# 		writer.add_histogram(name,param,step)
			#loss
			writer.add_scalar('loss',loss.item(),step)
			writer.add_scalar('loss_rec',loss_rec.item(),step)
			writer.add_scalar('loss_trans',loss_trans.item(),step)
			writer.add_scalar('loss_commit',loss_commit.item(),step)

		if not os.path.exists(os.path.join(args.log_dir, 'model')):
			os.makedirs(os.path.join(args.log_dir, 'model'))
		torch.save(model.state_dict(),os.path.join(args.log_dir,'model','epoch{}.pkl'.format(i)))
	writer.close()

