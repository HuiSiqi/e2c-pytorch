import torch
import os
import json
from .e2c import compute_loss
import types
from matplotlib import pyplot as plt
import imageio
def load_model(m,dir,filename):
	dicr_dir = os.path.join(dir,'model',filename)
	m.load_state_dict(torch.load(dicr_dir))

class DataEnc(json.JSONEncoder):
	def default(self, o):
		if isinstance(o,types.FunctionType):
			return o.__name__

def save_img(img,dir,file):
	if not os.path.exists(dir):
		os.makedirs(dir)
	plt.imsave(os.path.join(dir,file), img.copy().squeeze())

def save_fig(fig,dir,file):
	if not os.path.exists(dir):
		os.makedirs(dir)
	fig.savefig(os.path.join(dir,file))

def img2gif(imgdir,gifname):
	images = []
	filenames = sorted((os.path.join(imgdir,fn) for fn in os.listdir(imgdir)
	                    if fn.endswith('.png')))
	for filename in filenames:
		images.append(imageio.imread(filename))
	imageio.mimsave(os.path.join(imgdir,gifname+'.gif'),images,duration=0.3)