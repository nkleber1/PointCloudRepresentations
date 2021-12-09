"""
PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space
This code is based on the implementation of Erik Wijmans
https://github.com/erikwijmans/Pointnet2_PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
from typing import List, Optional, Tuple
import pytorch_lightning as pl


def build_shared_mlp(mlp_spec: List[int], bn: bool = True):
    layers = []
    for i in range(1, len(mlp_spec)):
        layers.append(
            nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn)
        )
        if bn:
            layers.append(nn.BatchNorm2d(mlp_spec[i]))
        layers.append(nn.ReLU(True))

    return nn.Sequential(*layers)


class PointnetSAModule(nn.Module):
    r"""Pointnet set abstrction layer
    Parameters
    ----------
    n_point : int
        Number of features
    radius : float
        Radius of ball
    n_sample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
        self, mlp, n_point=None, radius=None, n_sample=None, bn=True, use_xyz=True):  # TODO what does use_xyz doe
        # type: (PointnetSAModule, List[int], int, float, int, bool, bool) -> None
        super(PointnetSAModule, self).__init__()

        self.n_point = n_point
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        self.groupers.append(
            pointnet2_utils.QueryAndGroup(radius, n_sample, use_xyz=use_xyz)
            if n_point is not None
            else pointnet2_utils.GroupAll(use_xyz)
        )
        if use_xyz:
            mlp[0] += 3
        self.mlps.append(build_shared_mlp(mlp, bn))

    def forward(
        self, xyz: torch.Tensor, features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = (
            pointnet2_utils.gather_operation(
                xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.n_point)
            )
            .transpose(1, 2)
            .contiguous()
            if self.n_point is not None
            else None
        )

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointNet2CudaEncoder(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.SA_modules = nn.ModuleList()

        self.SA_modules.append(
            PointnetSAModule(
                n_point=512,
                radius=0.2,
                n_sample=64,
                mlp=[3, 64, 64, 128],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                n_point=128,
                radius=0.4,
                n_sample=64,
                mlp=[128, 128, 128, 256],
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024]
            )
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, args.feat_dims, bias=False),
            nn.BatchNorm1d(args.feat_dims)
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

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
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.fc_layer(features.squeeze(-1))

