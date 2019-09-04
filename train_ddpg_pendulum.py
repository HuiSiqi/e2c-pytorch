import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from ddpg import utils,env_wrap,ddpg, evaluator
from e2c import vqe2c,e2c,vae,datasets
from e2c.utils import load_model
from tqdm import trange
from copy import deepcopy
from tensorboardX import SummaryWriter
import os

# Hyperparameters


parser = argparse.ArgumentParser(description='train vae')

log = parser.add_argument_group('logger')
train = parser.add_argument_group('train')
model = parser.add_argument_group('model')
data = parser.add_argument_group('data')
env = parser.add_argument_group('env')

parser.add_argument(
	'--gpu-ids',
	type=int,
	default=[0,1,2,3],
	nargs='+',
	help='GPUs to use [-1 CPU only] (default: -1)')

train.add_argument('--step',type=int,default=6000000,metavar='TS',help='training steps')
train.add_argument('--lr',type=float,default=1e-3,metavar='lr',help='learning rate')
train.add_argument('--tau',type=float,default=0.005,metavar='TAU',help='target_net learning rate')
train.add_argument('--gamma',type=float,default=0.99,metavar='GAMMA',help='bellman equation param')
train.add_argument('--mu',type=float,default=0.1,metavar='MU',help='action sigma')
train.add_argument('--eval-inter',type=int,default=2000,metavar='Inter',help='evaluation interval')
# train.add_argument('--loss',default=vqe2c.compute_loss,metavar='LOSS',help='loss function')

data.add_argument('--bs',type=int,default=256,metavar='BS',help='training batch size')
data.add_argument('--ml',type=int,default=100000,metavar='ML',help='replay buffer size')
data.add_argument('--warmup',type=int,default=1000,metavar='WU',help='the number of ml at which begins to train')

env.add_argument('--epl',type=int,default=500,help='episode length')

log.add_argument('--log-dir',type=str,default='/home/pikey/Data/e2c/log/pendulum/ddpg',help='log directory')

model.add_argument('--h_dim',type=int,default=2,metavar='dz',help='latent_space_dimension')
model.add_argument('--topic_num',type=int,default=10,metavar='dz',help='latent_space_dimension')

def main():
	#todo init plugins
	args = parser.parse_args()
	writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'train_result'))
	memory = utils.ReplayBuffer(max_size=args.ml)
	eval = evaluator.Evaluator(10,args.eval_inter,args.log_dir,args.epl)
	#env
	env = gym.make('Pendulum-v0')
	f_z_o = vqe2c.VQE2C(datasets.GymPendulumDatasetV2.width*datasets.GymPendulumDatasetV2.height,2,1,4)
	load_model(f_z_o,'/home/pikey/Data/e2c/vqe2c-log','epoch499.pkl')
	env = env_wrap.Latent_ObservationEnv(env_wrap.NormalizedEnv(env),2,f_z_o)
	#agent
	agent = ddpg.DDPG(2,200,env.action_space,args.gamma,args.tau,args.mu)

	#todo collect warmup
	done = True
	ep_len = 0
	print('-----------collect data--------------')
	for i in trange(args.warmup):
		if done:
			s = deepcopy(env.reset())

		a = agent.random_action()
		s_prime, r, done, info = env.step([a])
		s_prime = deepcopy(s_prime)
		memory.add((s, a, r, s_prime, done))
		if ep_len>args.epl-1:
			done=True
		ep_len+=1
		s = deepcopy(s_prime)

	#todo train
	done = True
	score = 0.0
	n_ep = 0
	for step in trange(args.step):
		if done:
			s = deepcopy(env.reset())

		a = agent.select_action(s,action_noise=True)
		s_prime, r, done, info = env.step(a)
		s_prime = deepcopy(s_prime)
		memory.add((s, a, r, s_prime, done))

		if ep_len > args.epl - 1:
			done = True
		ep_len += 1
		score += r
		s = deepcopy(s_prime)

		if step%(args.ml/100) == 0 and step != 0:
			agent.eval(train=True)
			for i in range(int(args.ml/args.bs/10)):
				q_loss,mu_loss = agent.train(memory.sample(args.bs))
				writer.add_scalar('q_loss',q_loss,step)
				writer.add_scalar('mu_loss',mu_loss,step)
				print('q_loss:{}, mu_loss:{}'.format(q_loss,mu_loss))
			agent.eval()
		#[optional]evaluation
		if step%args.eval_inter==0 and step != 0:
			eval(env,lambda x:agent.select_action(x),debug=True)
			agent.save(args.log_dir)
		if done and step != 0:
			n_ep+=1
			score = 0.0

	env.close()


if __name__ == '__main__':
	main()