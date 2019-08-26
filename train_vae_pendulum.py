from e2c import vae,datasets
import argparse
from torch.utils.data import DataLoader,Sampler
from torch import optim
import torch
import os
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import cv2

parser = argparse.ArgumentParser(description='train vae')

env1 = parser.add_argument_group('environment1')
env2 = parser.add_argument_group('environment2')
log = parser.add_argument_group('logger')
train = parser.add_argument_group('train')
model = parser.add_argument_group('model')

parser.add_argument(
	'--gpu-ids',
	type=int,
	default=[0,1,2,3],
	nargs='+',
	help='GPUs to use [-1 CPU only] (default: -1)')

parser.add_argument(
	'--model-params',
	type=tuple,
	default = (),
	# default=(1,3),
	help='(dim_in,feature_dim,topic_num,topic_choose_num,action_num)')

#parser args
env1.add_argument('--se1',type=tuple,default=(0,100),metavar='SE1',help='env1 start end states')
env1.add_argument('--ser1',type=tuple,default=(-1,1),metavar='SER1',help='env1 start end reward')

env2.add_argument('--se2',type=tuple,default=(200,300),metavar='SE2',help='env2 start end states')
env2.add_argument('--ser2',type=tuple,default=(1,-1),metavar='SER2',help='env2 start end reward')

train.add_argument('--uold',type=int,default=1,metavar='UO',help='old model update intervals')
train.add_argument('--epoch',type=int,default=50,metavar='TS',help='training steps')
train.add_argument('--bs',type=int,default=256,metavar='BS',help='training batch size')
train.add_argument('--lr',type=float,default=1e-3,metavar='lr',help='learning rate')
train.add_argument('--loss',default=vae.compute_loss,metavar='LOSS',help='loss function')

log.add_argument('--log-dir',type=str,default='./log',help='log directory')

model.add_argument('--z-dim',type=int,default=2,metavar='dz',help='latent_space_dimension')

def get_args():
	return parser.parse_args()


if __name__ == '__main__':
	torch.manual_seed(1)
	args = get_args()
	writer = SummaryWriter(log_dir=os.path.join(args.log_dir,'train_result'))

	#todo data preparation
	dataset = datasets.GymPendulumDatasetV2(dir='dataset/pendulum')
	dataloader = DataLoader(dataset,args.bs,shuffle=True,
                        num_workers=16,drop_last=True)

	model = vae.VAE(datasets.GymPendulumDatasetV2.height*datasets.GymPendulumDatasetV2.width,dim_z=args.z_dim)

	opt = optim.Adam(model.parameters(),lr=args.lr)

	step= 0
	for i in range(args.epoch):
		j = 0
		for before,a,after in dataloader:
			step+=1
			j+=1
			before_reconstruction = model(before.view(args.bs,-1))
			loss_rec,loss_kl  =vae.compute_loss(before_reconstruction,before.view(args.bs,-1),
			                                    model.z_mean,model.z_logsigma,mse=True)
			loss = loss_rec+loss_kl
			opt.zero_grad()
			loss.backward()
			opt.step()
			print('loss:{}'.format(loss.item()))

			after_reconstruction = model(after.view(args.bs, -1))

			loss_rec, loss_kl = vae.compute_loss(after_reconstruction, after.view(args.bs, -1),
			                                     model.z_mean,model.z_logsigma, mse=True)
			loss = loss_rec + loss_kl
			opt.zero_grad()
			loss.backward()
			opt.step()

			#todo log
			print('loss:{}'.format(loss.item()))
			#save img
			img = after_reconstruction[0].detach().view(1,datasets.GymPendulumDatasetV2.height,datasets.GymPendulumDatasetV2.width)\
				.detach().cpu().numpy()
			# cv2.imshow('rec',img.transpose(1,2,0))
			# cv2.imshow('raw',after[0].numpy().transpose(1,2,0))
			# cv2.waitKey(1)
			rec_file = os.path.join(args.log_dir,'img','epoch{:03}_batch{:05d}.jpg'.format(i, j))
			if not os.path.exists(os.path.join(args.log_dir,'img')):
				os.makedirs(os.path.join(args.log_dir,'img'))
			plt.imsave(rec_file, img.copy().squeeze())
			#log
			#img
			writer.add_image('raw',after[0].numpy()
			                 ,step)
			writer.add_image('rec',img,step)
			#weight histogram
			for _, (name,param) in enumerate(model.named_parameters()):
				if 'bn' not in name:
					writer.add_histogram(name,param,step)
			#loss
			writer.add_scalar('loss',loss.item(),step)
		if not os.path.exists(os.path.join(args.log_dir, 'model')):
			os.makedirs(os.path.join(args.log_dir, 'model'))
		torch.save(model.state_dict(),os.path.join(args.log_dir,'model','param.pkl'))

	writer.close()

