"""
FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation
This code is based on the implementation of An Tao (ta19@mails.tsinghua.edu.cn)
"""


import torch
import torch.nn as nn
import numpy as np
import math
import itertools
from abc import ABC, abstractmethod
# TODO Make usable for all 1D, 2D, and 3D


class BaseFoldDecoder(nn.Module):
    def __init__(self, args):
        super(BaseFoldDecoder, self).__init__()
        self.args = args
        if '3d' in args.dataset:
            self.output_dim = 3
        else:
            self.output_dim = 2
        if args.shape == '1d':
            self.grid_dim = 1
        else:
            self.grid_dim = 2

    def build_grid(self, batch_size):
        n_points = self.args.num_points
        if self.args.shape == 'plane':
            n = int(math.sqrt(n_points))
            x = np.linspace(0, 1, n)
            y = np.linspace(0, 1, n)
            points = np.array(list(itertools.product(x, y)))
            points = points.transpose(1, 0)
        elif self.args.shape == '1d':
            points = np.linspace(0, 1, n_points)
            points = points[np.newaxis, ...]
        elif self.args.shape == 'diagonal':
            xy = np.linspace(0, 1, n_points)
            points = np.vstack((xy, xy))
        elif self.args.shape == 'circle':
            x = list()
            y = list()
            for i in range(n_points):
                i = 2 / n_points * i * math.pi
                sin = math.sin(i) / 2 + 0.5
                cos = math.cos(i) / 2 + 0.5
                x.append(sin)
                y.append(cos)
            points = np.array([x, y])
        elif self.args.shape == 'square':
            n_points = int(n_points / 4)
            x = np.linspace(0, 1, n_points + 2)
            x = x[1:n_points+1]
            x0 = np.zeros(n_points)
            x1 = np.ones(n_points)
            e0 = np.vstack((x0, x))
            e1 = np.vstack((x, x1))
            x = np.flip(x)
            e2 = np.vstack((x1, x))
            e3 = np.vstack((x, x0))
            points = np.hstack((e0, e1, e2, e3))
        elif self.args.shape == 'little_square':
            n_points = int(n_points / 4)
            x = np.linspace(0.25, 0.75, n_points + 2)
            x = x[1:n_points+1]
            x0 = np.zeros(n_points)
            x1 = np.ones(n_points)
            e0 = np.vstack((x0, x))
            e1 = np.vstack((x, x1))
            x = np.flip(x)
            e2 = np.vstack((x1, x))
            e3 = np.vstack((x, x0))
            points = np.hstack((e0, e1, e2, e3))
        elif self.args.shape == 'gaussian':
            x = np.random.normal(loc=0.5, scale=0.2, size=n_points * 2)
            points = np.reshape(x, (2, -1))
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
        points = torch.tensor(points)
        return points.float()

    def forward(self, x):
        x = x.transpose(1, 2).repeat(1, 1, self.args.num_points)  # (batch_size, feat_dims, num_points)
        points = self.build_grid(x.shape[0])   # (batch_size, [1,2,3], num_points)
        if x.get_device() != -1:
            points = points.cuda(x.get_device())
        cat1 = torch.cat((x, points), dim=1)  # (batch_size, feat_dims+[1,2,3], num_points)
        folding_result1 = self.folding1(cat1)  # (batch_size, [2,3], num_points)
        cat2 = torch.cat((x, folding_result1), dim=1)  # (batch_size, 514, num_points)
        folding_result2 = self.folding2(cat2)  # (batch_size, [2,3], num_points)
        return folding_result2.transpose(1, 2)  # (batch_size, num_points, [2,3])


class FoldDecoder(BaseFoldDecoder):
    def __init__(self, args):
        super(FoldDecoder, self).__init__(args)
        self.folding1 = nn.Sequential(
            nn.Conv1d(args.feat_dims + self.grid_dim, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, self.output_dim, 1),
        )
        self.folding2 = nn.Sequential(
            nn.Conv1d(args.feat_dims + self.output_dim, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, self.output_dim, 1),
        )


class FoldDecoderS(BaseFoldDecoder):
    def __init__(self, args):
        super(FoldDecoderS, self).__init__(args)
        self.folding1 = nn.Sequential(
            nn.Conv1d(args.feat_dims + self.grid_dim, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, self.output_dim, 1),
        )
        self.folding2 = nn.Sequential(
            nn.Conv1d(args.feat_dims + self.output_dim, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, self.output_dim, 1),
        )


class FoldSingleDecoder(BaseFoldDecoder):
    def __init__(self, args):
        super(FoldSingleDecoder, self).__init__(args)
        self.folding1 = nn.Sequential(
            nn.Conv1d(args.feat_dims + self.grid_dim, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, self.output_dim, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2).repeat(1, 1, self.args.num_points)
        points = self.build_grid(x.shape[0])
        if x.get_device() != -1:
            points = points.cuda(x.get_device())
        cat1 = torch.cat((x, points), dim=1)
        folding_result1 = self.folding1(cat1)
        return folding_result1.transpose(1, 2)

