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
            eval_file = 'data/eval_data/lidar_512.npy'
        elif args.dataset == 'uniform_density':
            file = 'data/train_data/uniform_density.npy'
            eval_file = 'data/eval_data/uniform_density_eval.npy'
        elif args.dataset == 'overfit':
            file = 'data/train_data/overfit.npy'
            eval_file = 'data/eval_data/easy_eval.npy'
        elif args.dataset == 'easy':
            file = 'data/train_data/easy.npy'
            eval_file = 'data/eval_data/easy_eval.npy'
        elif args.dataset == 'medium':
            file = 'data/train_data/medium.npy'
            eval_file = 'data/eval_data/medium.npy'
        np_data = np.load(file)
        np_eval_data = np.load(eval_file)[:10]
        self.data = torch.from_numpy(np_data)
        self.eval_data = torch.from_numpy(np_eval_data)
        self.test = torch.unsqueeze(self.data[9], 0)
        self.unseen_test = torch.unsqueeze(self.eval_data[0], 0)
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

    def get_eval_data(self):
        return self.eval_data
