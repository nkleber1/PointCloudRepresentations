"""
FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation
This code is based on the implementation of An Tao (ta19@mails.tsinghua.edu.cn)
"""

import os
import sys
import time
import shutil
import json
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter
from .model import ReconstructionNet
from .dataset import PointCloudDataset
from .utils import Logger


class Reconstruction(object):
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.snapshot_interval = args.snapshot_interval
        self.snapshot_pic = True  # TODO add to args
        self.no_cuda = args.no_cuda
        self.model_path = args.model_path

        # create exp directory
        file = [f for f in args.model_path.split('/')]
        if args.exp_name != None:
            self.experiment_id = "Reconstruct_" + args.exp_name
        elif file[-2] == 'models':
            self.experiment_id = file[-3]
        else:
            self.experiment_id = "Reconstruct" + time.strftime('%m%d%H%M%S')
        snapshot_root = f'logging/snapshot/{str(self.experiment_id)}'
        tensorboard_root = f'logging//tensorboard/{str(self.experiment_id)}'
        self.save_dir = os.path.join(snapshot_root, 'models/')
        self.plot_dir = os.path.join(snapshot_root, 'plot/')
        self.tboard_dir = tensorboard_root

        # check arguments
        if self.model_path == '':
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            elif args.overwrite:
                shutil.rmtree(self.save_dir)
                os.makedirs(self.save_dir)
            else:
                choose = input("Remove " + self.save_dir + " ? (y/n)")
                if choose == "y":
                    shutil.rmtree(self.save_dir)
                    os.makedirs(self.save_dir)
                else:
                    sys.exit(0)
            if not os.path.exists(self.tboard_dir):
                os.makedirs(self.tboard_dir)
            else:
                shutil.rmtree(self.tboard_dir)
                os.makedirs(self.tboard_dir)
            if not os.path.exists(self.plot_dir):
                os.makedirs(self.plot_dir)
            else:
                shutil.rmtree(self.plot_dir)
                os.makedirs(self.plot_dir)
        sys.stdout = Logger(os.path.join(snapshot_root, 'log.txt'))
        self.writer = SummaryWriter(log_dir=self.tboard_dir)

        jason_dir = os.path.join(snapshot_root, 'args.json')
        with open(jason_dir, 'w') as fp:
            json.dump(vars(args), fp, indent=5)

        # print args
        print(str(args))

        # get gpu id  # TODO Whats this? Does this work for Batch System?
        gids = ''.join(args.gpu.split())
        self.gpu_ids = [int(gid) for gid in gids.split(',')]
        self.first_gpu = self.gpu_ids[0]

        # generate dataset  # TODO use better dataset
        self.train_dataset = PointCloudDataset(args)
        self.test_data = self.train_dataset.test

        # self.train_dataset = Dataset(
        #     root=args.dataset_root,
        #     dataset_name=args.dataset,
        #     split='all',
        #     num_points=args.num_points,
        #     random_translate=args.use_translate,
        #     random_rotate=True,
        #     random_jitter=args.use_jitter
        # )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers
        )
        print("Training set size:", self.train_loader.dataset.__len__())

        # initialize model
        self.model = ReconstructionNet(args)
        if self.model_path != '':
            self._load_pretrain(args.model_path)

        # load model to gpu
        if not self.no_cuda:
            if len(self.gpu_ids) != 1:  # multiple gpus
                self.model = torch.nn.DataParallel(self.model.cuda(self.first_gpu), self.gpu_ids)
            else:
                self.model = self.model.cuda(self.gpu_ids[0])

        # initialize optimizer
        self.parameter = self.model.parameters()
        self.optimizer = optim.Adam(self.parameter, lr=0.0001 * 16 / args.batch_size, betas=(0.9, 0.999),
                                    weight_decay=1e-6)

    def run(self):
        self.train_hist = {
            'loss': [],
            'loss_per_point': [],
            'per_epoch_time': [],
            'total_time': []
        }
        best_loss = 1000000000
        print('Training start!!')
        start_time = time.time()
        self.model.train()
        if self.model_path != '':
            start_epoch = self.model_path[-10:-4]
            start_epoch = start_epoch.split('_')[-1]
            start_epoch = int(start_epoch)
        else:
            start_epoch = 0
        for epoch in range(start_epoch, self.epochs):
            loss = self.train_epoch(epoch)

            # save snapeshot
            if (epoch + 1) % self.snapshot_interval == 0:
                self._snapshot(epoch + 1)
                if loss < best_loss:
                    best_loss = loss
                    self._snapshot('best')

            # save tensorboard
            if self.writer:
                self.writer.add_scalar('Train Loss', self.train_hist['loss'][-1], epoch)
                self.writer.add_scalar('Train Loss (per point)', self.train_hist['loss_per_point'][-1], epoch)
                self.writer.add_scalar('Learning Rate', self._get_lr(), epoch)

        # finish all epoch
        self._snapshot(epoch + 1)
        if loss < best_loss:
            best_loss = loss
            self._snapshot('best')
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epochs, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

    def train_epoch(self, epoch):
        epoch_start_time = time.time()
        loss_buf = []
        loss_per_point_buf = []
        num_batch = int(len(self.train_loader.dataset) / self.batch_size)
        for iter, pts in enumerate(self.train_loader):
            if not self.no_cuda:
                pts = pts.cuda(self.first_gpu)

            # forward
            self.optimizer.zero_grad()
            output, _ = self.model(pts)

            # loss
            if len(self.gpu_ids) != 1:  # multiple gpus
                loss = self.model.module.get_loss(pts, output)
            else:
                loss = self.model.get_loss(pts, output)

            # backward
            loss.backward()
            self.optimizer.step()
            loss_buf.append(loss.detach().cpu().numpy())
            loss_per_point_buf.append(loss.detach().cpu().numpy()/self.args.num_points/self.args.batch_size)

        # finish one epoch
        epoch_time = time.time() - epoch_start_time
        self.train_hist['per_epoch_time'].append(epoch_time)
        self.train_hist['loss'].append(np.mean(loss_buf))
        self.train_hist['loss_per_point'].append(np.mean(loss_per_point_buf))
        print(f'Epoch {epoch + 1}: Loss {np.mean(loss_per_point_buf)}, time {epoch_time:.4f}s')
        return np.mean(loss_buf)

    def _snapshot(self, epoch):
        state_dict = self.model.state_dict()
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key[:6] == 'module':
                name = key[7:]  # remove 'module.'
            else:
                name = key
            new_state_dict[name] = val
        save_dir = os.path.join(self.save_dir, self.dataset_name)
        torch.save(new_state_dict, save_dir + "_" + str(epoch) + '.pkl')
        print(f"Save model to {save_dir}_{str(epoch)}.pkl")
        if self.snapshot_pic:
            self._plot_reconstruction(epoch)

    def _plot_reconstruction(self, epoch):
        x = self.test_data
        self.model.eval()
        embedding = self.model.encoder.forward(x)
        reconstruction = self.model.decoder.forward(embedding)
        # make plot
        fig = plt.figure()
        subplot = fig.add_subplot(111)
        # plot ground truth
        x = x.detach().numpy()
        subplot.scatter(x[0, :, 0], x[0, :, 1], s=10, c='b', marker="s", label='true')
        # plot reconstruction
        reconstruction = reconstruction.detach().numpy()
        subplot.scatter(reconstruction[0, :, 0], reconstruction[0, :, 1], s=10, c='r', marker="o", label='reconstruction')
        plt.legend(loc='upper left')
        # save plot
        save_dir = os.path.join(self.plot_dir, self.dataset_name)
        save_dir = save_dir + "_" + str(epoch) + '.png'
        plt.savefig(save_dir)
        print(f"Save model to {save_dir}")
        self.model.train()

    def _load_pretrain(self, pretrain):
        state_dict = torch.load(pretrain, map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key[:6] == 'module':
                name = key[7:]  # remove 'module.'
            else:
                name = key
            new_state_dict[name] = val
        self.model.load_state_dict(new_state_dict)
        print(f"Load model from {pretrain}")

    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']
