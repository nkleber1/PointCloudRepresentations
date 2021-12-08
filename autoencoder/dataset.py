import torch
import numpy as np


def rotate_pointcloud(pointcloud):
    theta = np.pi * 2 * np.random.choice(24) / 24
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pointcloud[:, [0, 1]] = pointcloud[:, [0, 1]].dot(rotation_matrix)  # random rotation (x,z)
    return pointcloud


class PointCloudDataset:
    def __init__(self, args):
        if args.dataset == 'lidar':
            file = 'data/train_data/lidar_512.npy'
        elif args.dataset == 'uniform_density':
            file = 'data/train_data/uniform_density_2048.npy'
        elif args.dataset == 'overfit':
            file = 'data/train_data/overfit.npy'
        elif args.dataset == 'easy':
            file = 'data/train_data/easy.npy'
        elif args.dataset == 'medium':
            file = 'data/train_data/medium.npy'
        np_data = np.load(file)
        self.data = torch.from_numpy(np_data)
        if args.dataset == 'lidar':
            self.test = torch.unsqueeze(self.data[44], 0)
            self.n_points = np_data.shape[1]
        else:
            self.test = torch.unsqueeze(self.data[9], 0)
            self.n_points = args.num_points
        self.test.float()
        self.n_clouds = np_data.shape[0]
        self.point_dim = np_data.shape[2]
        self.rotate = args.rotate

    def __getitem__(self, index):
        point_set = self.data[index][:self.n_points].float()
        if self.rotate:
            point_set = rotate_pointcloud(point_set)
        return point_set

    def __len__(self):
        return self.n_clouds