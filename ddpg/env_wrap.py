
import gym
from e2c import datasets
from gym import spaces
import numpy as np
import torch
from PIL import Image
from collections import deque

# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """
    def __init__(self,env):
        super(NormalizedEnv, self).__init__(env)
        self.act_k = (self.action_space.high - self.action_space.low) / 2.
        self.act_b = (self.action_space.high + self.action_space.low) / 2.
        self.act_k_inv = 2. / (self.action_space.high - self.action_space.low)
    def action(self, action):
        return self.act_k * action + self.act_b
    def reverse_action(self, action):
        return self.act_k_inv * (action - self.act_b)

class Latent_ObservationEnv(gym.ObservationWrapper):
    def __init__(self,env,window_size,dec):
        super(Latent_ObservationEnv, self).__init__(env)
        self.dec = dec
        self.dec.eval()
        high = np.array([np.inf, np.inf])
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.init_state(window_size)
        self.window_size = window_size
    def init_state(self,window_size):
        self.state = deque(maxlen=window_size)
        zero = np.zeros((100,100,3))
        for i in range(window_size):
            self.state.append(zero.copy())
    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        self.init_state(self.window_size)
        return self.observation(ob)
    def observation(self,ob):
        self.state.append(self.env.render(mode='rgb_array'))
        img = np.concatenate(list(self.state), axis=1)
        img = Image.fromarray(np.uint8(img))
        img = datasets.GymPendulumDatasetV2._process_image(img).view(1,-1)
        with torch.no_grad():
            ob = self.dec.latent_embeddings(img).squeeze().cpu().numpy()
        return ob



