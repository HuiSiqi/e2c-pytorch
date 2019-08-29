from e2c import datasets,vae,utils,e2c
import numpy as np
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn
from tqdm import tqdm

def visualize(model,dataloader,gpu='cpu'):
	dim_z = model.encoder.dim_out
	x = np.empty(shape=(0,2))
	z = np.empty(shape=(0,dim_z))
	print('______test______')
	for image,state in tqdm(dataloader):
	# 	#todo state sample
		state  = np.array(state.numpy())
		x = np.vstack((x,state))
		#todo latent sample
		image = image.to(gpu)
		state = model.latent_embeddings(image.view(image.shape[0],-1))
		state = state.detach().cpu().numpy()
		z = np.vstack((z,state))
	# plot(x)
	fig_z = plot(z)
	return fig_z

def plot(data,s=3):
	#todo draw
	figure = plt.figure()
	if min(data.shape)==2:
		axes = figure.add_subplot(1,1,1)
		axes.scatter(x=data[:,0],y=data[:,1],c=np.linspace(0,1,data.shape[0]),cmap='rainbow',s=s)
	elif min(data.shape)==3:
		axes = Axes3D(figure)
		axes.scatter(xs=data[:, 0], ys=data[:, 1],zs=data[:, 2],c=np.linspace(0, 1, data.shape[0]), cmap='rainbow', s=s)

	return figure

if __name__ == '__main__':
	BS = 256
	dset = datasets.GymPendulumDatasetV2_visual('/home/pikey/Data/e2c/visual_dataset')
	dloader = DataLoader(dset, BS, shuffle=False)
	#
	model = e2c.E2C(dim_in=datasets.GymPendulumDatasetV2_visual.width * datasets.GymPendulumDatasetV2_visual.height,
	                dim_z=2,dim_u=1)
	utils.load_model(model, '/home/pikey/Data/e2c/e2clog','epoch499.pkl')

	visualize(model,dloader)