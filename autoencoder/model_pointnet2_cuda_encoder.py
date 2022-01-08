"""
PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space
This code is based on the implementation of Erik Wijmans
https://github.com/erikwijmans/Pointnet2_PyTorch
"""
import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetSAModule
from typing import List, Optional, Tuple
import pytorch_lightning as pl


def _break_up_pc(pc):
    xyz = pc[..., 0:3].contiguous()
    features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

    return xyz, features


class PointNet2CudaEncoder(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        output_dim = args.feat_dims
        if not args.no_vae:
            output_dim = output_dim * 2

        self.SA_modules = nn.ModuleList()

        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=64,
                mlp=[-1, 32, 32, 64],  # [-1, 64, 64, 128],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[64-1, 64, 64, 128],  # [128-1, 128, 128, 256],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[128-1, 256, 512, 1024]  # [256-1, 256, 512, 1024]
            )
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, output_dim, bias=False),
            nn.BatchNorm1d(output_dim)
        )

    def forward(self, pointcloud):
        """
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = _break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        features = self.fc_layer(features.squeeze(-1))
        # features = features.transpose(2, 1)
        return torch.unsqueeze(features, 1)

