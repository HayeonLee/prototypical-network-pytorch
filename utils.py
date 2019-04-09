import os
import shutil
import time
import pprint

import torch
import torch.nn as nn
import torch.distributions.kl as kl
import torch.distributions as dist
Normal = dist.Normal


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(path):
    if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.mkdir(path)
    else:
        os.mkdir(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


def dot_metric(a, b):
    return torch.mm(a, b.t())


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


class Proto_KL(object):
    def__init__(self):
        super(Combine, self).__init__()
        self.qc_x = qc_x().to('cuda')

    def kl_div_metric(self, qry_mu, qry_sigma, spt_mu, spt_sigma):
        n = qry_mu.shape[0] # a.size: [75, 1600]
        m = qry_sigma.shape[0] # b.size: [5, 1600]   

        qry_mu = qry_mu.unsqueeze(1).expand(n, m, -1) # a.size: [75, 5, 1600]
        qry_sigma = qry_sigma.unsqueeze(1).expand(n, m, -1) # a.size: [75, 5, 1600]
        spt_mu = spt_mu.unsqueeze(0).expand(n, m, -1) # b.size: [75, 5, 1600]
        spt_sigma = spt_sigma.unsqueeze(0).expand(n, m, -1) # b.size: [75, 5, 1600]
        qry = Normal(qry_mu, qry_sigma)
        spt = Normal(spt_mu, spt_sigma)
        logits = - kl.kl_divergence(spt, qry).sum(dim=2)
        return logits

    def disc(self, qry, proto):
        # qry.size: [75, 1600], proto.size: [5, 1600]

        # support points
        # cX_mu.size: [5, 128], cx_mu.size: [25,128]
        spt_mu, spt_sigma, _, _, _ = self.qc_x(proto) 
        # query points
        # qry_mu: [75, 128], qry_sigma: [75, 128]
        _, _, qry_mu, qry_sigma, _ = self.qc_x(xqry) 

        # spt = Normal(spt_mu, spt_sigma) #[n, cdim]
        # qry = Normal(qry_mu, qry_sigma) #[nq, cdim]

        logits = kl_div_metric(qry_mu, qry_sigma, spt_mu, spt_sigma)

        return logits 

    def train(self):
        self.qc_x.train()

    def eval(self):
        self.qc_x.eval()      


class postpool(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(postpool, self).__init__()
        self.mu = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.mu.weight)
        self.sigma = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.sigma.weight)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x_mu = self.mu(x)
        x_sigma = self.sigma(x)
        x_sigma = self.softplus(x_sigma)
        return x_mu, x_sigma


class qc_x(nn.Module):
    def __init__(self, ):
        super(qc_x, self).__init__()
        self.conv2d = nn.Conv2d(16, 32, kernel_size=3, padding=2)
        nn.init.xavier_uniform_(self.conv2d.weight)
        self.bn = nn.BatchNorm2d(32, momentum=0.01)
        self.elu = nn.ELU()
        self.maxpool = nn.MaxPool2d(2)
        self.fc = nn.Linear(1152, 256) # hdim: 256
        self.postpool = postpool(256, 128)
        self.way = 5
        self.shot = 5
        self.hdim = 256

    def forward(self, x):
        x = x.view(-1, 16, 10, 10)
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.maxpool(x) # x size: [25, 32, 6, 6]
        x = x.view(x.size(0), -1) # x size: [25, 1152]

        x = self.fc(x) #[25, 256]

        # Set
        X = x.view(self.way, -1, self.hdim) # [5, 5, 256]
        X = torch.sum(X, dim=1) # X size: [5, 256]

        X_mu, X_sigma = self.postpool(X) # X_mu, X_sigma: [5, 128], [5, 128]
        x_mu, x_sigma = self.postpool(x) # x_mu, x_sigma: [25, 128], [25, 128]

        return X_mu, X_sigma, x_mu, x_sigma, X


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def l2_loss(pred, label):
    return ((pred - label)**2).sum() / len(pred) / 2

