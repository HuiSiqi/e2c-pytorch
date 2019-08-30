from e2c import datasets,vae,utils,e2c
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn
from tqdm import tqdm
import os

def visualize(model,dataloader,gpu='cpu'):
	dim_z = model.encoder.dim_out
	x = np.empty(shape=(0,2))
	z = np.empty(shape=(0,dim_z))
	cls =[]
	model.eval()
	print('______test______')
	for image,state in dataloader:
	# 	#todo state sample
		state  = np.array(state.numpy())
		x = np.vstack((x,state))

		#todo latent sample
		image = image.to(gpu)
		state = model.latent_embeddings(image.view(image.shape[0],-1))
		quantize,indices = 	model.topic(state)
		state = state.detach().cpu().numpy()
		indices = indices.cpu().numpy().tolist()
		z = np.vstack((z,state))
		cls+=indices
	cls = np.array(cls) / model.quantize.n_e
	c = model.quantize.embed.cpu().numpy()

	model.train()
	fig_x_cls = plot(x,c_indices=cls)
	fig_z = plot(z)
	fig_z_cls = plot(z,c_indices=cls)
	fig_topic_center = plot(c)
	return fig_x_cls,fig_z,fig_z_cls,fig_topic_center

def plot(data,c_indices = None,s=3):
	#todo draw
	figure = plt.figure()

	#todo set color
	if type(c_indices) == type(None):
		c = np.linspace(0, 1, data.shape[0])
	else:
		c = c_indices
	if min(data.shape)==2:
		axes = figure.add_subplot(1,1,1)
		axes.scatter(x=data[:,0],y=data[:,1],c=c,cmap='rainbow',s=s)
	elif min(data.shape)==3:
		axes = Axes3D(figure)
		axes.scatter(xs=data[:, 0], ys=data[:, 1],zs=data[:, 2],c=c, cmap='rainbow', s=s)

	return figure


