import torch
from torch import nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
	def __init__(self, num_features, eps=1e-5, affine=True):
		super(LayerNorm, self).__init__()
		self.num_features = num_features
		self.affine = affine
		self.eps = eps
		self.shape = (-1,self.num_features)

		if self.affine:
			gamma = torch.randn(num_features).uniform_().requires_grad_(False)
			beta = torch.zeros(num_features).requires_grad_(False)
			self.register_buffer('gamma',gamma)
			self.register_buffer('beta',beta)

	def forward(self, x):
		x = x.view(*self.shape)
		mean = x.mean(0).view(*self.shape).detach()
		std = x.std(0).view(*self.shape).detach()

		y = (x - mean) / (std + self.eps)
		if self.affine:
			y = self.gamma.clone().view(*self.shape) * y + self.beta.clone().view(*self.shape)
			self.gamma.add_(1e-3,std.view(-1)-self.gamma)
			self.beta.add_(1e-3,mean.view(-1)-self.beta)
		return y


class Actor(nn.Module):
	def __init__(self, hidden_size, num_inputs, action_space):
		super(Actor, self).__init__()
		self.action_space = action_space
		num_outputs = action_space.shape[0]

		self.linear1 = nn.Linear(num_inputs, hidden_size)
		self.ln1 = nn.BatchNorm1d(hidden_size)

		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.ln2 = nn.BatchNorm1d(hidden_size)

		self.mu = nn.Linear(hidden_size, num_outputs)
		self.mu.weight.data.mul_(0.1)
		self.mu.bias.data.mul_(0.1)

	def forward(self, inputs):
		x = inputs
		x = self.linear1(x)
		x = self.ln1(x)
		x = F.relu(x)
		x = self.linear2(x)
		x = self.ln2(x)
		x = F.relu(x)
		mu = torch.tanh(self.mu(x))
		return mu

class Critic(nn.Module):
	def __init__(self, hidden_size, num_inputs, action_space):
		super(Critic, self).__init__()
		self.action_space = action_space
		num_outputs = action_space.shape[0]

		self.linear1 = nn.Linear(num_inputs, hidden_size)
		self.ln1 = nn.BatchNorm1d(hidden_size)

		self.linear2 = nn.Linear(num_outputs, hidden_size)
		self.ln2 = nn.BatchNorm1d(hidden_size)

		self.linear3 = nn.Linear(hidden_size, hidden_size)
		self.ln3 = nn.BatchNorm1d(hidden_size)

		self.V = nn.Linear(hidden_size, 1)
		self.V.weight.data.mul_(0.1)
		self.V.bias.data.mul_(0.1)

	def forward(self, inputs, actions):
		x = inputs
		x = self.linear1(x)
		x = self.ln1(x)
		a = self.linear2(actions)
		a = self.ln2(a)
		x = F.relu(x+a)
		x = self.linear3(x)
		x = self.ln3(x)
		x = F.relu(x)
		V = self.V(x)
		return V