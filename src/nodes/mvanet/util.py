# These are different from BEN
from torch import nn


def make_cbr(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU())


def make_cbg(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.GELU())
