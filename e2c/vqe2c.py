import torch
from torch import nn,distributions
from .losses import binary_crossentropy

class VQE2C(nn.Module):
    def __init__(self, dim_in, dim_z, dim_u,topic_num, config='vq-pendulum'):
        super(VQE2C, self).__init__()
        enc, trans,quantize, dec = load_config(config)
        self.encoder = enc(dim_in, dim_z)
        self.decoder = dec(dim_z, dim_in)
        self.trans = trans(dim_z, dim_u)
        self.quantize = quantize(topic_num,dim_z)
        self.prior = distributions.Normal(0, 1)
    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def transition(self, z, Qz, u):
        return self.trans(z, Qz, u)

    def reparam(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        self.z_mean = mean
        self.z_sigma = std
        eps = self.prior.sample()
        if std.data.is_cuda:
            eps.to(std.device)

        # return eps.mul(std).add_(mean), NormalDistribution(mean, std, torch.log(std))
        cov = []
        for _ in std:
            cov.append(torch.diag(_))
        cov = torch.stack(cov,dim=0)
        return eps.mul(std).add_(mean), distributions.MultivariateNormal(mean, cov)

    def forward(self, x, action, x_next):
        mean, logvar = self.encode(x)
        mean_next, logvar_next = self.encode(x_next)

        z,self.Qz = self.reparam(mean, logvar)
        z_next,self.Qz_next = self.reparam(mean_next, logvar_next)

        self.x_dec = self.decode(z)
        self.x_next_dec = self.decode(z_next)

        self.z_next_pred, self.Qz_next_pred = self.transition(z, self.Qz, action)
        self.x_next_pred_dec = self.decode(self.z_next_pred)

        self.z_diff = self.quantize(z)
        self.z_next_diff = self.quantize(z_next)

        return self.x_next_pred_dec

    def latent_embeddings(self, x):
        return self.encode(x)[0]


    def predict(self, X, U):
        mean, logvar = self.encode(X)
        z, Qz = self.reparam(mean, logvar)
        z_next_pred, Qz_next_pred = self.transition(z, Qz, U)
        return self.decode(z_next_pred)


def compute_loss(x_dec, x_next_pred_dec, x_next_dec,x, x_next,
                 Qz, Qz_next_pred,
                 Qz_next,z_diff,z_next_diff,mse=False):
    # Reconstruction losses
    if mse:
        x_reconst_loss = (x_dec - x).pow(2).sum(dim=1)
        x_next_reconst_loss = (x_next_dec - x_next).pow(2).sum(dim=1)
        x_next_pre_reconst_loss = (x_next_pred_dec - x_next).pow(2).sum(dim=1)
    else:
        x_reconst_loss = -binary_crossentropy(x, x_dec).sum(dim=1)
        x_next_reconst_loss = -binary_crossentropy(x_next, x_next_dec).sum(dim=1)
        x_next_pre_reconst_loss = -binary_crossentropy(x_next, x_next_pred_dec).sum(dim=1)


    prior = distributions.MultivariateNormal(torch.zeros_like(Qz.mean[0]),torch.diag(torch.ones_like(Qz.mean[0])))
    z_KLD = distributions.kl_divergence(Qz,prior)
    z_next_KLD = distributions.kl_divergence(Qz_next,prior)

    # ELBO
    bound_loss = x_reconst_loss.add(x_next_reconst_loss).add(z_KLD).add(z_next_KLD)
    trans_loss = distributions.kl_divergence(Qz_next_pred, Qz_next).add(x_next_pre_reconst_loss)

    # commitment loss
    commit_loss = z_diff+z_next_diff
    return bound_loss.mean()/2, trans_loss.mean(), commit_loss/2

from .configs import load_config