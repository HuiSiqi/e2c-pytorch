from e2c import datasets,vae,utils,e2c
import numpy as np
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn
from tqdm import tqdm

def visualize(model,dataloader):
	dim_z = model.encoder.dim_out
	x = np.empty(shape=(0,2))
	z = np.empty(shape=(0,dim_z))
	for image,state in tqdm(dataloader):
	# 	#todo state sample
		state  = np.array(state.numpy())
		x = np.vstack((x,state))
		np.save('x.npy',x)
		#todo latent sample
		state = model.latent_embeddings(image.view(image.shape[0],-1))
		state = state.detach().numpy()
		z = np.vstack((z,state))
		np.save('z.npy',z)
	x = np.load('x.npy')
	z = np.load('z.npy')
	plot(x)
	plot(z)

def plot(data):
	#todo draw
	figure = plt.figure()
	if min(data.shape)==2:
		axes = figure.add_subplot(1,1,1)
		axes.scatter(x=data[:,0],y=data[:,1],c=np.linspace(0,1,data.shape[0]),cmap='rainbow',s=3)
		plt.show()
	elif min(data.shape)==3:
		axes = Axes3D(figure)
		axes.scatter(xs=data[:, 0], ys=data[:, 1],zs=data[:, 2],c=np.linspace(0, 1, data.shape[0]), cmap='rainbow', s=3)
		plt.show()


if __name__ == '__main__':
	BS = 256
	dset = datasets.GymPendulumDatasetV2_visual('visual_dataset')
	dloader = DataLoader(dset, BS, shuffle=False)
	#
	model = e2c.E2C(dim_in=datasets.GymPendulumDatasetV2_visual.width * datasets.GymPendulumDatasetV2_visual.height,
	                dim_z=2,dim_u=1)
	utils.load_model(model, '/home/pikey/Data/e2c/e2clog','epoch49.pkl')

	visualize(model,dloader)