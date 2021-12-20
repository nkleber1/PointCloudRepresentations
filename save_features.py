import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from autoencoder import ReconstructionNet, PointCloudDataset
from autoencoder.dataset import PointCloudEvalDataset


def get_parser():
    parser = argparse.ArgumentParser(description='Unsupervised Point Cloud Feature Learning')
    parser.add_argument('--model_path', type=str, default='Reconstruct_foldingnet_easy_eval_16_99_01_rot',
                        metavar='N', help='Path to load model')
    parser.add_argument('--encoder', type=str, default='graph', metavar='N',
                        choices=['graph', 'pointnet++', 'pointnet2cuda', 'pointnet', 'dense'],
                        help='Encoder architecture used, [graph, pointnet++, pointnet, dense]')
    parser.add_argument('--decoder', type=str, default='fold', metavar='N',
                        choices=['fold', 'upsampling', 'dense'],
                        help='Decoder architecture used, [fold, upsampling, dense]')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout rate')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='Num of points to use')
    parser.add_argument('--feat_dims', type=int, default=16, metavar='N',
                        help='Number of dims for feature')
    parser.add_argument('--k', type=int, default=64, metavar='N',
                        help='Num of nearest neighbors to use for KNN')
    parser.add_argument('--shape', type=str, default='square', metavar='N',
                        choices=['1d', 'diagonal', 'circle', 'square', 'little_square', 'gaussian'],
                        help='Shape of points to input decoder, [1d, diagonal, circle, square, gaussian]')
    parser.add_argument('--pooling', type=str, default='avg', metavar='N',
                        choices=['avg', 'max'],
                        help='Pooling type used for PointNet, [avg, max]')
    parser.add_argument('--dataset', type=str, default='uniform', metavar='N',
                        choices=['lidar', 'uniform', 'overfit', 'easy', 'medium'],
                        help='Encoder to use, [lidar, uniform, overfit, easy, medium]')
    parser.add_argument('--no_rotate', action='store_false',
                        help='rotate point clouds during training')
    args = parser.parse_args()
    return args


def main():
    args = get_parser()
    # generate model path
    model_path = 'logging/snapshot/{}/eval_models/'.format(args.model_path)
    model_path = os.path.join(model_path, os.listdir(model_path)[-1])
    # initialize model
    model = ReconstructionNet(args)
    model.load_pretrain(model_path)
    model.eval()
    # generate dataset
    dataset = PointCloudEvalDataset(args)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    # make save dir
    save_dir = 'data/feat/train_{}_{}/'.format(args.dataset, args.feat_dims)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, pts in enumerate(loader):
        feat = model.encoder(pts)
        np_feat = feat.cpu().detach().numpy()
        name_file = 'point_feat_{}'.format(i + 1)
        path = os.path.join(save_dir, name_file)
        np.save(path, np_feat)



if __name__ == '__main__':
    main()