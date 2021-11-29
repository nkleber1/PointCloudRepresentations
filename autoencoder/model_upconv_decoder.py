"""
This code is inspired by the tensorflow implementation by Charles R. Qi
(https://github.com/charlesq34/pointnet-autoencoder)
"""

import argparse
import torch.nn as nn
import torch.nn.functional as F
from point_clouds.embedding.autoencoder import DenseEncoder
from point_clouds.embedding.autoencoder.dataset import PointCloudDataset

class UpConvDecoder(nn.Module):
    def __init__(self, args):
        super(UpConvDecoder, self).__init__()
        self.m = args.num_points

        # FC Decoder # TODO do we need that
        # self.fc1 = nn.Linear(in_features=args.feat_dims, out_features=512)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.fc2 = nn.Linear(in_features=512, out_features=512)
        # self.bn2 = nn.BatchNorm1d(512)
        # self.fc3 = nn.Linear(in_features=512, out_features=1024 * 2)

        # UPCONV Decoder
        self.upconv1 = nn.Conv2d(1, 512, kernel_size=(2, 2), padding='valid')
        self.bn1 = nn.BatchNorm2d(512)
        self.upconv2 = nn.Conv2d(512, 256, kernel_size=(3, 3), padding='valid')
        self.bn2 = nn.BatchNorm2d(256)
        self.upconv3 = nn.Conv2d(256, 256, kernel_size=(4, 4), stride=(2,2), padding='valid')
        self.bn3 = nn.BatchNorm2d(256)
        self.upconv4 = nn.Conv2d(256, 128, kernel_size=(5, 5), stride=(3, 3), padding='valid')
        self.bn4 = nn.BatchNorm2d(128)
        self.upconv5 = nn.Conv2d(128, 2, kernel_size=(1, 1), padding='valid')


    def forward(self, feat):
        B, _, _ = feat.shape
        # FC Decoder
        # feat = feat.squeeze()
        # pts = F.relu(self.bn1(self.fc1(feat)))
        # pts = F.relu(self.bn2(self.fc2(pts)))
        # pts = self.fc3(pts)
        # pc_fc = pts.reshape(B, 2, 1024)

        # UPCONV Decoder
        pts = feat.reshape(B, 1, 2, -1)
        pts = F.relu(self.bn1(self.upconv1(pts)))
        print(pts.shape)
        pts = F.relu(self.bn2(self.upconv2(pts)))
        pts = F.relu(self.bn3(self.upconv3(pts)))
        pts = F.relu(self.bn4(self.upconv4(pts)))
        pts = self.upconv4(pts)
        print(pts.shape)
        return pts.reshape(B, 2, -1)


def get_parser():
    parser = argparse.ArgumentParser(description='Unsupervised Point Cloud Feature Learning')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='Num of points to use')
    parser.add_argument('--feat_dims', type=int, default=128, metavar='N',
                        help='Number of dims for feature ')
    parser.add_argument('--dataset', type=str, default='uniform_density', metavar='N',
                        choices=['lidar', 'uniform_density', 'regular_distances'],
                        help='Encoder to use, [lidar, uniform_density, regular_distances]')
    parser.add_argument('--rotate', action='store_true',
                        help='rotate point clouds during training')
    args = parser.parse_args()
    return args

args = get_parser()
data = PointCloudDataset(args)[:5]
print(data.shape)
encoder = DenseEncoder(args)
decoder = UpConvDecoder(args)
feat = encoder(data)
print(feat.shape)
re = decoder(feat)
print(re.shape)

