"""
FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation
This code is based on the implementation of An Tao (ta19@mails.tsinghua.edu.cn)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoder import GraphEncoder


class GraphDoubleEncoder(nn.Module):
    def __init__(self, args):
        super(GraphDoubleEncoder, self).__init__()
        output_dim = args.feat_dims
        if not args.no_vae:
            output_dim = output_dim * 2

        self.encoder1 = GraphEncoder(args, args.k)
        self.encoder2 = GraphEncoder(args, args.k / 4)
        self.mlp = nn.Sequential(
            nn.Conv1d(output_dim * 2, output_dim, 1),
            nn.ReLU(),
        )

    def forward(self, pts):
        feature1 = self.encoder1(pts)
        feature2 = self.encoder1(pts)
        feature = torch.cat((feature1, feature2), dim=2)
        feature = feature.transpose(2, 1)
        return self.mlp(feature)
