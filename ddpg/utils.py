import random,collections
import torch
import numpy as np

def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
def prLightPurple(prt): print("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): print("\033[98m {}\033[00m" .format(prt))

class ReplayBuffer(object):
	def __init__(self, max_size=1e6):
		self.storage = []
		self.max_size = max_size
		self.ptr = 0

	def add(self, data):
		if len(self.storage) == self.max_size:
			self.storage[int(self.ptr)] = data
			self.ptr = (self.ptr + 1) % self.max_size
		else:
			self.storage.append(data)

	def sample(self, batch_size):
		ind = np.random.randint(0, len(self.storage), size=batch_size)
		s, a, r, s_, d = [], [], [], [], []

		for i in ind:
			S, A, R, S_, D = self.storage[i]
			s.append(S)
			a.append(np.array([A], copy=False))
			r.append([R])
			s_.append(S_)
			d.append([D])

		return torch.from_numpy(np.array(s)).type(torch.FloatTensor), torch.from_numpy(np.array(a)).view(-1,1).type(torch.FloatTensor), \
		       torch.from_numpy(np.array(r)).view(-1,1).type(torch.FloatTensor), torch.from_numpy(np.array(s_)).type(torch.FloatTensor), \
		       torch.from_numpy(np.array(d)).view(-1, 1).type(torch.FloatTensor)

	def size(self):
		return len(self.storage)

class OrnsteinUhlenbeckNoise:
	def __init__(self, mu):
		self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
		self.mu = mu
		self.reset()

	def __call__(self):
		x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
				self.sigma * np.sqrt(self.dt) * np.random.normal()
		self.x_prev = x
		return x

	def reset(self):
		self.x_prev = np.zeros_like(self.mu)

def soft_update(target, source, tau):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(param.data)