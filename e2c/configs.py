"""
Configuration for the encoder, decoder, transition
for different tasks. Use load_config to find the proper
set of configuration.
"""
import torch
from torch import nn,distributions
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, enc, dim_in, dim_out):
        super(Encoder, self).__init__()
        self.m = enc
        self.dim_int = dim_in
        self.dim_out = dim_out

    def forward(self, x):
        return self.m(x).chunk(2, dim=1)

class Decoder(nn.Module):
    def __init__(self, dec, dim_in, dim_out):
        super(Decoder, self).__init__()
        self.m = dec
        self.dim_in = dim_in
        self.dim_out = dim_out

    def forward(self, z):
        return self.m(z)

class Transition(nn.Module):
    def __init__(self, trans, dim_z, dim_u):
        super(Transition, self).__init__()
        self.trans = trans
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.fc_B = nn.Linear(dim_z, dim_z * dim_u)
        self.fc_o = nn.Linear(dim_z, dim_z)

    def forward(self, h, Q, u):
        batch_size = h.size()[0]
        v, r = self.trans(h).chunk(2, dim=1)
        v1 = v.unsqueeze(2)
        rT = r.unsqueeze(1)
        I = torch.eye(self.dim_z).repeat(batch_size, 1, 1)
        if rT.is_cuda:
            I = I.to(rT.device)
        A = I.add(v1.bmm(rT))

        B = self.fc_B(h).view(-1, self.dim_z, self.dim_u)
        o = self.fc_o(h)

        # need to compute the parameters for distributions
        # as well as for the samples
        u = u.unsqueeze(2)

        d = A.bmm(Q.mean.unsqueeze(2)).add(B.bmm(u)).add(o.unsqueeze(2)).squeeze(2)
        sample = A.bmm(h.unsqueeze(2)).add(B.bmm(u)).add(o.unsqueeze(2)).squeeze(2)

        z_cov = Q.covariance_matrix
        z_next_cov = A.bmm(z_cov).bmm(A.transpose(1,2))
        # return sample, NormalDistribution(d, Q.sigma, Q.logsigma, v=v, r=r)
        return sample, distributions.MultivariateNormal(d,z_next_cov)


class Quantize(nn.Module):
    def __init__(self, n_e, e_dim, decay=0.99, eps=1e-5):
        super(Quantize, self).__init__()

        self.e_dim = e_dim
        self.n_e = n_e
        self.decay = decay
        self.eps = eps

        embed = F.normalize(torch.randn(n_e, e_dim))
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.ones(n_e))
        self.register_buffer('embed_to_avg', embed.clone())

    def em_center(self, indices):
        assert indices.dim() == 1
        assert indices.dtype == torch.int64
        return F.embedding(indices, self.embed)

    def similarity(self, x):
        flatten = x.view(-1, self.e_dim)
        sim = (self.embed / (self.embed.norm(dim=1, keepdim=True) ** 2)).mm(flatten.transpose(0, 1)).transpose(0, 1)
        return sim

    def hard_assign(self, x):
        x = x.view(-1,self.e_dim)
        sim = self.similarity(x)
        _, embed_ind = (sim).max(1)
        quantize = self.em_center(embed_ind)

        return quantize,embed_ind

    def hard_assign_train(self,x):
        x = x.view(-1, self.e_dim)
        sim = self.similarity(x)
        _, embed_ind = (sim).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_e).type(x.dtype)
        quantize = self.em_center(embed_ind)
        if self.training:
            with torch.no_grad():
                self.cluster_size.mul_(self.decay).add_(
                    1 - self.decay, embed_onehot.sum(0)
                )
                embed_sum = embed_onehot.transpose(0, 1).mm(x)
                self.embed_to_avg.mul_(self.decay).add_(1 - self.decay, embed_sum)
                n = self.cluster_size.sum()

                cluster_size = (self.cluster_size + self.eps) / (n + self.n_e * self.eps) * n

                embed_normalized = self.embed_to_avg / cluster_size.unsqueeze(1)
                self.embed.copy_(embed_normalized)
        diff = (quantize.detach() - x).pow(2).mean()
        return diff

    def forward(self, x):
        if self.training:
            diff = self.hard_assign_train(x)
            return diff
        else:
            return self.hard_assign(x)


class PlaneEncoder(Encoder):
    def __init__(self, dim_in, dim_out):
        m = nn.Sequential(
            nn.Linear(dim_in, 150),
            nn.BatchNorm1d(150),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.BatchNorm1d(150),
            nn.ReLU(),
            nn.Linear(150, 150),
            nn.BatchNorm1d(150),
            nn.ReLU(),
            nn.Linear(150, dim_out*2)
        )
        super(PlaneEncoder, self).__init__(m, dim_in, dim_out)


class PlaneDecoder(Decoder):
    def __init__(self, dim_in, dim_out):
        m = nn.Sequential(
            nn.Linear(dim_in, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, dim_out),
            nn.BatchNorm1d(dim_out),
            nn.Sigmoid()
        )
        super(PlaneDecoder, self).__init__(m, dim_in, dim_out)


class PlaneTransition(Transition):
    def __init__(self, dim_z, dim_u):
        trans = nn.Sequential(
            nn.Linear(dim_z, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, dim_z*2)
        )
        super(PlaneTransition, self).__init__(trans, dim_z, dim_u)


class PendulumEncoder(Encoder):
    def __init__(self, dim_in, dim_out):
        m = nn.ModuleList([
            torch.nn.Linear(dim_in, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            torch.nn.Linear(800, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Linear(800, 2 * dim_out)
        ])
        super(PendulumEncoder, self).__init__(m, dim_in, dim_out)

    def forward(self, x):
        for l in self.m:
            x = l(x)
        return x.chunk(2, dim=1)


class PendulumDecoder(Decoder):
    def __init__(self, dim_in, dim_out):
        m = nn.ModuleList([
            torch.nn.Linear(dim_in, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            torch.nn.Linear(800, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Linear(800, dim_out),
            nn.Sigmoid()
        ])
        super(PendulumDecoder, self).__init__(m, dim_in, dim_out)

    def forward(self, z):
        for l in self.m:
            z = l(z)
        return z


class PendulumTransition(Transition):
    def __init__(self, dim_z, dim_u):
        trans = nn.Sequential(
            nn.Linear(dim_z, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, dim_z * 2),
            nn.BatchNorm1d(dim_z * 2),
            nn.Sigmoid() # Added to prevent nan
        )
        super(PendulumTransition, self).__init__(trans, dim_z, dim_u)


_CONFIG_MAP = {
    'plane': (PlaneEncoder, PlaneTransition, PlaneDecoder),
    'pendulum': (PendulumEncoder, PendulumTransition, PendulumDecoder),
    'vq-pendulum':(PendulumEncoder,PendulumTransition,Quantize,PendulumDecoder)
}


def load_config(name):
    """Load a particular configuration
    Returns:
    (encoder, transition, decoder) A tuple containing class constructors
    """
    if name not in _CONFIG_MAP.keys():
        raise ValueError("Unknown config: %s", name)
    return _CONFIG_MAP[name]

__all__ = ['load_config']
