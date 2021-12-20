import argparse
import os
import torch
import matplotlib.pyplot as plt
from autoencoder import ReconstructionNet, PointCloudDataset
from autoencoder.dataset import PointCloudEvalDataset


def get_parser():
    parser = argparse.ArgumentParser(description='Unsupervised Point Cloud Feature Learning')
    parser.add_argument('--model_path', type=str, default='Reconstruct_graph_dense_uniform_16_98_05',  # 'Reconstruct_foldingnet_easy_eval_16_99_01_rot',
                        metavar='N', help='Path to load model')
    parser.add_argument('--encoder', type=str, default='graph', metavar='N',
                        choices=['graph', 'pointnet++', 'pointnet2cuda', 'pointnet', 'dense'],
                        help='Encoder architecture used, [graph, pointnet++, pointnet, dense]')
    parser.add_argument('--decoder', type=str, default='dense', metavar='N',
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
    parser.add_argument('--dataset', type=str, default='easy', metavar='N',
                        choices=['lidar', 'uniform', 'overfit', 'easy', 'medium'],
                        help='Encoder to use, [lidar, uniform, overfit, easy, medium]')
    parser.add_argument('--no_rotate', action='store_false',
                        help='rotate point clouds during training')
    args = parser.parse_args()
    return args


def plot(gt, reconstruction):
    # make plot
    fig = plt.figure()
    subplot = fig.add_subplot(111)
    # plot ground truth
    gt = gt.detach().cpu().numpy()
    subplot.scatter(gt[0, :, 0], gt[0, :, 1], s=10, c='b', marker="s", label='true')
    # plot reconstruction
    reconstruction = reconstruction.cpu().detach().numpy()
    subplot.scatter(reconstruction[0, :, 0], reconstruction[0, :, 1], s=10, c='r', marker="o",
                    label='reconstruction')
    plt.legend(loc='upper left')
    plt.show()


def main():
    args = get_parser()
    # generate model path
    model_path = 'logging/snapshot/{}/models/'.format(args.model_path)
    model_path = os.path.join(model_path, os.listdir(model_path)[-1])
    # initialize model
    model = ReconstructionNet(args)
    model.load_pretrain(model_path)
    model.eval()
    # generate dataset
    dataset = PointCloudDataset(args)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for _, pts in enumerate(loader):
        output, _ = model(pts)
        plot(pts, output)



if __name__ == '__main__':
    main()