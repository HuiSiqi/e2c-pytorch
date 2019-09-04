from .config import *
from .utils import *
from torch.optim import Adam
import torch
import os
class DDPG(object):
	def __init__(self,state_dim,hidden_dim,action_space,gamma=0.99,tau=0.001,mu=0):
		#todo save hp
		self.state_dim = state_dim
		self.action_space = action_space
		self.gamma = gamma
		self.tau = tau

		self.actor = Actor(hidden_dim,state_dim,action_space)
		self.target_actor = Actor(hidden_dim,state_dim,action_space)
		self.critic = Critic(hidden_dim,state_dim,action_space)
		self.target_critic = Critic(hidden_dim,state_dim,action_space)

		self.actor_optim = Adam(self.actor.parameters(), lr=1e-3)
		self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

		hard_update(self.target_actor,self.actor)
		hard_update(self.target_critic,self.critic)

		self.random = OrnsteinUhlenbeckNoise(mu)
		self.eval()

	def select_action(self, state, action_noise=False):
		state = torch.from_numpy(state).unsqueeze(0)
		mu = self.actor((state))
		mu = mu
		if action_noise:
			mu += torch.Tensor([self.random()]).view(1,-1)

		return mu.clamp(-1, 1).cpu().detach().item()

	def random_action(self):
		return random.uniform(-1,1)

	def train(self,batch):
		s, a, r, s_prime, done_mask = batch

		target = r + done_mask*self.gamma * self.target_critic(s_prime, self.target_actor(s_prime))
		q_loss = F.mse_loss(self.critic(s, a), target.detach())
		self.critic_optim.zero_grad()
		q_loss.backward()
		self.critic_optim.step()

		mu_loss = -self.critic(s, self.actor(s)).mean()  # That's all for the policy loss.
		self.actor_optim.zero_grad()
		mu_loss.backward()
		self.actor_optim.step()

		soft_update(self.target_actor, self.actor, self.tau)
		soft_update(self.target_critic, self.critic, self.tau)

		return q_loss.item(),mu_loss.item()

	def eval(self,train=False):
		if train:
			self.target_actor.train()
			self.actor.train()
			self.critic.train()
			self.target_critic.train()
		else:
			self.target_critic.eval()
			self.critic.eval()
			self.target_actor.eval()
			self.actor.eval()

	def save(self,dir):
		dir = os.path.join(dir,'model')
		if not os.path.exists(dir):
			os.makedirs(dir)
		torch.save(self.target_critic.state_dict(),os.path.join(dir,'critic.pkl'))
		torch.save(self.target_actor.state_dict(),os.path.join(dir,'actor.pkl'))