from e2c import datasets

a = datasets.GymPendulumDatasetV2.sample(50000,'/home/pikey/Data/e2c/dataset/pendulum',1,num_shards=50)
a  = datasets.GymPendulumDatasetV2('/home/pikey/Data/e2c/dataset/pendulum')