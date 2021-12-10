"""
This code is based on the implementation of An Tao (ta19@mails.tsinghua.edu.cn)
"""

import argparse
import torch

from autoencoder import Reconstruction


def get_parser():
    parser = argparse.ArgumentParser(description='Unsupervised Point Cloud Feature Learning')
    parser.add_argument('--exp_name', type=str, default=None, metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrites data from previous experiment with same name')
    parser.add_argument('--encoder', type=str, default='graph', metavar='N',
                        choices=['graph', 'pointnet++', 'pointnet2cuda', 'pointnet', 'dense'],
                        help='Encoder architecture used, [graph, pointnet++, pointnet, dense]')
    parser.add_argument('--decoder', type=str, default='fold', metavar='N',
                        choices=['fold', 'upsampling', 'dense'],
                        help='Decoder architecture used, [fold, upsampling, dense]')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout rate')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning Rate')
    parser.add_argument('--lr_scheduler_gamma', type=float, default=0.99,
                        help='multiplies lr by lr_scheduler_gamma all lr_scheduler_step')
    parser.add_argument('--lr_scheduler_steps', type=int, default=100,
                        help='multiplies lr by lr_scheduler_gamma all lr_scheduler_step')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='Num of points to use')
    parser.add_argument('--feat_dims', type=int, default=32, metavar='N',
                        help='Number of dims for feature')
    parser.add_argument('--k', type=int, default=64, metavar='N',
                        help='Num of nearest neighbors to use for KNN')
    parser.add_argument('--batch_size', type=int, default=64, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=1000000, metavar='N',
                        help='Number of episode to train ')
    parser.add_argument('--no_snapshot_plot', action='store_false',
                        help='save a plot after snapshot')
    parser.add_argument('--snapshot_interval', type=int, default=100, metavar='N',
                        help='Save snapshot interval')
    parser.add_argument('--no_eval_plot', action='store_false',
                        help='save a plot after eval')
    parser.add_argument('--eval_interval', type=int, default=100, metavar='N',
                        help='Evaluation interval')
    parser.add_argument('--shape', type=str, default='square', metavar='N',
                        choices=['1d', 'diagonal', 'circle', 'square', 'little_square', 'gaussian'],
                        help='Shape of points to input decoder, [1d, diagonal, circle, square, gaussian]')
    parser.add_argument('--pooling', type=str, default='avg', metavar='N',
                        choices=['avg', 'max'],
                        help='Pooling type used for PointNet, [avg, max]')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate the model')
    parser.add_argument('--model_path', type=str, default='',
                        metavar='N', help='Path to load model')
    parser.add_argument('--dataset', type=str, default='overfit', metavar='N',
                        choices=['lidar', 'uniform_density', 'overfit', 'easy', 'medium'],
                        help='Encoder to use, [lidar, uniform_density, overfit, easy, medium]')
    parser.add_argument('--rotate', action='store_true',
                        help='rotate point clouds during training')
    parser.add_argument('--gpu', type=str, help='Id of gpu device to be used', default='0')
    # parser.add_argument('--no_cuda', action='store_true',
    #                      help='Enables CUDA training')
    parser.add_argument('--no_cuda', type=int, default=True,
                        help='Enables CUDA training')
    parser.add_argument('--workers', type=int, help='Number of data loading workers', default=0)

    args = parser.parse_args()
    return args


def main():
    args = get_parser()
    c = torch.cuda
    if c.is_available():
        c.empty_cache()
        print('CUDA Information:')
        print('-- is_available: \t', c.is_available())
        print('-- current_device: \t', c.current_device())
        print('-- device: \t\t', c.device(0))
        print('-- device_count: \t', c.device_count())
        print('-- get_device_name: \t', c.get_device_name(0))
        print('-- no_cuda: \t\t', args.no_cuda)
    reconstruction = Reconstruction(args)
    reconstruction.run()


if __name__ == '__main__':
    main()


