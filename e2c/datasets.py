import glob
import os
from os import path

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import gym
import json
from datetime import datetime
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from .tf_e2c.plane_data2 import T, num_t
from skimage.transform import resize
from skimage.color import rgb2gray
from tqdm import trange, tqdm
import pickle


class PendulumData(Dataset):
	def __init__(self, root, split):
		if split not in ['train', 'test', 'all']:
			raise ValueError

		dir = os.path.join(root, split)
		filenames = glob.glob(os.path.join(dir, '*.png'))

		if split == 'all':
			filenames = glob.glob(os.path.join(root, 'train/*.png'))
			filenames.extend(glob.glob(os.path.join(root, 'test/*.png')))

		filenames = sorted(
			filenames, key=lambda x: int(os.path.basename(x).split('.')[0]))

		images = []

		for f in filenames:
			img = plt.imread(f)
			img[img != 1] = 0
			images.append(resize(rgb2gray(img), [48, 48], mode='constant'))

		self.images = np.array(images, dtype=np.float32)
		self.images = self.images.reshape([len(images), 48, 48, 1])

		action_filename = os.path.join(root, 'actions.txt')

		with open(action_filename) as infile:
			actions = np.array([float(l) for l in infile.readlines()])

		self.actions = actions[:len(self.images)].astype(np.float32)
		self.actions = self.actions.reshape(len(actions), 1)

	def __len__(self):
		return len(self.actions) - 1

	def __getitem__(self, index):
		return self.images[index], self.actions[index], self.images[index]


class PlaneDataset(Dataset):
	def __init__(self, planedata):
		self.planedata = planedata

	def __len__(self):
		return T * num_t  # Total number of samples

	def __getitem__(self, index):
		index = np.random.randint(0, num_t)  # Sample any one of them
		t = np.random.randint(0, T - 1)
		x = np.array(self.planedata.getX(index, t))
		x_next = np.array(self.planedata.getX(index, t + 1))
		u = np.copy(self.planedata.U[index, t, :])
		return x, u, x_next


class GymPendulumDataset(Dataset):
	"""Dataset definition for the Gym Pendulum task"""
	width = 40
	height = 40
	action_dim = 1
	"""Sample from the OpenAI Gym environment, requires a patched version of gym"""

	def __init__(self, filename):
		_data = np.load(filename)
		self.X0 = np.copy(_data['X0'])  # Copy to memory, otherwise it's slow.
		self.X1 = np.copy(_data['X1'])
		self.U = np.copy(_data['U'])
		_data.close()

	def __len__(self):
		return len(self.X0)

	def __getitem__(self, index):
		return self.X0[index], self.U[index], self.X1[index]

	@classmethod
	def all_states(cls):
		_env = gym.make('Pendulum-v0').env
		width = GymPendulumDataset.width
		height = GymPendulumDataset.height
		X = np.zeros((360, width, height))

		for i in range(360):
			th = i / 360. * 2 * np.pi
			state = _env.render_state(th)
			X[i, :, :] = resize(rgb2gray(state), (width, height), mode='reflect')
		_env.close()
		_env.viewer.close()
		return X

	@classmethod
	def sample_trajectories(self, sample_size, step_size=1, apply_control=True):
		_env = gym.make('Pendulum-v0').env
		X0 = np.zeros((sample_size, 500, 500, 3), dtype=np.uint8)
		U = np.zeros((sample_size, 1), dtype=np.float32)
		X1 = np.zeros((sample_size, 500, 500, 3), dtype=np.uint8)
		for i in range(sample_size):
			th = np.random.uniform(0, np.pi * 2)
			# thdot = np.random.uniform(-8, 8)
			thdot = 0
			state = np.array([th, thdot])
			initial = state
			# apply the same control over a few timesteps
			if apply_control:
				u = np.random.uniform(-2, 2, size=(1,))
			else:
				u = np.zeros((1,))
			for _ in range(step_size):
				state = _env.step_from_state(state, u)

			X0[i, :, :, :] = _env.render_state(initial[0])
			U[i, :] = u
			X1[i, :, :, :] = _env.render_state(state[0])
		_env.viewer.close()
		return X0, U, X1


class GymPendulumDatasetV2(Dataset):
	width = 40 * 2
	height = 40
	action_dim = 1

	def __init__(self, dir):
		self.dir = dir
		with open(os.path.join(dir, 'data.json')) as f:
			self._data = json.load(f)
		self._process()

	def __len__(self):
		return len(self._data['samples'])

	def __getitem__(self, index):
		return self._processed[index]

	@staticmethod
	def _process_image(img):
		return ToTensor()((img.convert('L').
						   resize((GymPendulumDatasetV2.width,
								   GymPendulumDatasetV2.height))))

	def _process(self):
		preprocessed_file = os.path.join(self.dir, 'processed.pkl')
		if not os.path.exists(preprocessed_file):
			processed = []
			for sample in tqdm(self._data['samples'], desc='processing data'):
				before = Image.open(os.path.join(self.dir, sample['before']))
				after = Image.open(os.path.join(self.dir, sample['after']))

				processed.append((self._process_image(before),
								  np.array(sample['control']),
				                  np.array(sample['avg_r']),
								  self._process_image(after)))

			with open(preprocessed_file, 'wb') as f:
				pickle.dump(processed, f)
			self._processed = processed
		else:
			with open(preprocessed_file, 'rb') as f:
				self._processed = pickle.load(f)

	@staticmethod
	def _render_state_fully_observed(env, state):
		env.reset(state=state)
		before1 = env.render(mode='rgb_array')
		env.step(np.array([0]))
		before2 = env.render(mode='rgb_array')
		return [before1,before2]

	@classmethod
	def sample(cls,sample_size, output_dir, step_size=1,
			   apply_control=True, num_shards=10):
		env = gym.make('Pendulum-v0').unwrapped
		env = pendulumwrap(env)
		assert sample_size % num_shards == 0

		samples = []

		if not path.exists(output_dir):
			os.makedirs(output_dir)

		for i in trange(sample_size):
			th = np.random.uniform(0, np.pi * 2)
			thdot = np.random.uniform(-8, 8)

			state = np.array([th, thdot])

			initial_state = state
			before1, before2 = GymPendulumDatasetV2._render_state_fully_observed(env, state)

			# apply the same control over a few timesteps
			if apply_control:
				u = np.random.uniform(-2, 2, size=(1,))
			else:
				u = np.zeros((1,))

			# state = env.step_from_state(state, u0)
			env.reset(state=state)

			total_r = 0
			for _ in range(step_size):
				_,r,_,_ = env.step(u)
				total_r+=r
			r = total_r/step_size


			after_state = env.state
			after1, after2 = GymPendulumDatasetV2._render_state_fully_observed(env, env.state)

			before = np.hstack((before1, before2))
			after = np.hstack((after1, after2))

			shard_no = i //(sample_size // num_shards)

			shard_path = path.join('{:03d}-of-{:03d}'.format(shard_no,num_shards))

			if not path.exists(path.join(output_dir, shard_path)):
				os.makedirs(path.join(output_dir, shard_path))

			before_file = path.join(shard_path, 'before-{:05d}.jpg'.format(i))
			plt.imsave(path.join(output_dir, before_file), before)

			after_file = path.join(shard_path, 'after-{:05d}.jpg'.format(i))
			plt.imsave(path.join(output_dir, after_file), after)

			samples.append({
				'before_state': initial_state.tolist(),
				'after_state': after_state.tolist(),
				'before': before_file,
				'after': after_file,
				'control': u.tolist(),
				'avg_r':[r],
			})

		with open(path.join(output_dir, 'data.json'), 'wt') as outfile:
			json.dump(
				{
					'metadata': {
						'num_samples': sample_size,
						'step_size': step_size,
						'apply_control': apply_control,
						'time_created': str(datetime.now()),
						'version': 2
					},
					'samples': samples
				}, outfile, indent=2)

		env.close()

#todo designed for visualization
class GymPendulumDatasetV2_visual(GymPendulumDatasetV2):

	def __init__(self,dir):
		super(GymPendulumDatasetV2_visual, self).__init__(dir)
	def _process(self):
		preprocessed_file = os.path.join(self.dir, 'processed.pkl')
		if not os.path.exists(preprocessed_file):
			processed = []
			for sample in tqdm(self._data['samples'], desc='processing data'):
				before = Image.open(os.path.join(self.dir, sample['img']))

				processed.append((self._process_image(before),
								  np.array(sample['state'])
				                  ))
			with open(preprocessed_file, 'wb') as f:
				pickle.dump(processed, f)
			self._processed = processed
		else:
			with open(preprocessed_file, 'rb') as f:
				self._processed = pickle.load(f)

	@classmethod
	def all_state(cls, output_dir,th_num,thdot_num):
		def theta_norm(th):
			return th/np.pi*180

		env = gym.make('Pendulum-v0').unwrapped
		env = pendulumwrap(env)

		samples = []
		if not path.exists(output_dir):
			os.makedirs(output_dir)
		for th in tqdm(np.linspace(0,np.pi*2,th_num)):
			th_dir = str(theta_norm(th))
			if not path.exists(os.path.join(output_dir,th_dir)):
				os.makedirs(os.path.join(output_dir,th_dir))
			for thdot in np.linspace(0,env.max_speed,int(thdot_num/2)):
				state = np.array([th, thdot])
				initial_state = state
				before1, before2 = GymPendulumDatasetV2._render_state_fully_observed(env, state)
				before = np.hstack((before1, before2))

				state_file = os.path.join(th_dir,'th{:03f}-thdot-{:03f}.jpg'.format(th, thdot))
				plt.imsave(path.join(output_dir,state_file), before)

				samples.append({
					'state': initial_state.tolist(),
					'img': state_file,
				})
		for th in tqdm(np.linspace(np.pi*2,0,th_num)):
			th_dir = str(theta_norm(th))
			if not path.exists(os.path.join(output_dir,th_dir)):
				os.makedirs(os.path.join(output_dir,th_dir))
			for thdot in np.linspace(0,-env.max_speed,int(thdot_num/2)):
				state = np.array([th, thdot])
				initial_state = state
				before1, before2 = GymPendulumDatasetV2._render_state_fully_observed(env, state)
				before = np.hstack((before1, before2))

				state_file = os.path.join(th_dir,'th{:03f}-thdot-{:03f}.jpg'.format(th, thdot))
				plt.imsave(path.join(output_dir,state_file), before)

				samples.append({
					'state': initial_state.tolist(),
					'img': state_file,
				})

		with open(path.join(output_dir, 'data.json'), 'wt') as outfile:
			json.dump(
				{
					'metadata': {
						'num_samples': th_num*thdot_num,
						'time_created': str(datetime.now()),
						'version': 2
					},
					'samples': samples
				}, outfile, indent=2)
		env.close()


class pendulumwrap(gym.Wrapper):
	def __init__(self,env):
		super(pendulumwrap, self).__init__(env)

	def reset(self, **kwargs):
		if 'state' in kwargs.keys():
			th,thdot = kwargs['state']
			th = np.clip(th,0,2*np.pi)
			thdot = np.clip(thdot,-self.env.max_speed,self.env.max_speed)
			self.env.state = np.array([th,thdot])
			self.env.last_u = None
			return self.env._get_obs()
		else:
			return self.env.reset(**kwargs)

	@property
	def state(self):
		return self.env.state