import torch
import numpy as np


def rotate_pointcloud(pointcloud):
    theta = np.pi * 2 * np.random.choice(24) / 24
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotation_matrix = torch.from_numpy(rotation_matrix).float()
    pointcloud = pointcloud.mm(rotation_matrix)  # random rotation (x,z)
    return pointcloud

class PointCloudDataset:
    def __init__(self, args):
        if args.dataset == 'lidar':
            file = 'data/train_data/lidar_512.npy'
        elif args.dataset == 'uniform':
            file = 'data/train_data/uniform.npy'
        elif args.dataset == 'overfit':
            file = 'data/train_data/overfit.npy'
        elif args.dataset == 'easy':
            file = 'data/train_data/easy.npy'
        elif args.dataset == 'medium':
            file = 'data/train_data/medium.npy'
        np_data = np.load(file)
        self.data = torch.from_numpy(np_data)
        self.n_clouds = np_data.shape[0]
        self.n_points = np_data.shape[1]
        self.point_dim = np_data.shape[2]
        self.rotate = args.rotate

    def __getitem__(self, index):
        point_set = self.data[index][:self.n_points].float()
        if not self.no_rotate:
            point_set = rotate_pointcloud(point_set)
        return point_set

    def __len__(self):
        return self.n_clouds

    def get_plot_data(self):
        plot_data = torch.unsqueeze(self.data[0], 0)
        plot_data.float()
        return plot_data


class PointCloudEvalDataset:
    def __init__(self, args):
        if args.dataset == 'lidar':
            file = 'data/eval_data/lidar_512_eval.npy'
        elif args.dataset == 'uniform':
            file = 'data/eval_data/uniform_eval.npy'
        elif args.dataset == 'overfit':
            file = 'data/eval_data/easy_eval.npy'
        elif args.dataset == 'easy':
            file = 'data/eval_data/easy_eval.npy'
        elif args.dataset == 'medium':
            file = 'data/eval_data/medium.npy'
        np_data = np.load(file)
        self.data = torch.from_numpy(np_data)
        self.n_clouds = np_data.shape[0]
        self.n_points = np_data.shape[1]
        self.point_dim = np_data.shape[2]

    def __getitem__(self, index):
        point_set = self.data[index][:self.n_points].float()
        return point_set

    def __len__(self):
        return self.n_clouds

    def get_plot_data(self):
        plot_data = torch.unsqueeze(self.data[0], 0)
        plot_data.float()
        return plot_data
