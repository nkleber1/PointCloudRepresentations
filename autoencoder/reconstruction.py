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
from torch.optim import lr_scheduler
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
        self.eval_interval = args.eval_interval
        self.snapshot_plot = not args.no_snapshot_plot
        self.eval_plot = not args.no_eval_plot
        self.no_cuda = args.no_cuda
        self.model_path = args.model_path

        if self.model_path[:12] == 'Reconstruct_':
            self.model_path = 'logging/snapshot/{}/models/latest'.format(self.model_path)
        if self.model_path[-6:] == 'latest':
            path = self.model_path[:-6]
            latest_model = os.listdir(path)[-2]
            self.model_path = os.path.join(path, latest_model)

        # create exp directory
        file = [f for f in self.model_path.split('/')]
        if args.exp_name != None:
            self.experiment_id = "Reconstruct_" + args.exp_name
        elif file[-2] == 'models':
            self.experiment_id = file[-3]
        else:
            self.experiment_id = "Reconstruct" + time.strftime('%m%d%H%M%S')
        snapshot_root = f'logging/snapshot/{str(self.experiment_id)}'
        tensorboard_root = f'logging//tensorboard/{str(self.experiment_id)}'
        self.model_dir = os.path.join(snapshot_root, 'models/')
        self.plot_dir = os.path.join(snapshot_root, 'plot/')
        self.eval_plot_dir = os.path.join(snapshot_root, 'eval_plot/')
        self.tboard_dir = tensorboard_root

        # check arguments
        if self.model_path == '':
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            elif args.overwrite:
                shutil.rmtree(self.model_dir)
                os.makedirs(self.model_dir)
            else:
                choose = input("Remove " + self.model_dir + " ? (y/n)")
                if choose == "y":
                    shutil.rmtree(self.model_dir)
                    os.makedirs(self.model_dir)
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
            if not os.path.exists(self.eval_plot_dir):
                os.makedirs(self.eval_plot_dir)
            else:
                shutil.rmtree(self.eval_plot_dir)
                os.makedirs(self.eval_plot_dir)
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

        # generate dataset
        self.dataset = PointCloudDataset(args)

        self.train_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers
        )
        print("Training set size:", self.train_loader.dataset.__len__())

        # initialize model
        self.model = ReconstructionNet(args)
        if self.model_path != '':
            self._load_pretrain(self.model_path)

        # load model to gpu
        if not self.no_cuda:
            if len(self.gpu_ids) != 1:  # multiple gpus
                self.model = torch.nn.DataParallel(self.model.cuda(self.first_gpu), self.gpu_ids)
            else:
                self.model = self.model.cuda(self.gpu_ids[0])

        # initialize optimizer
        self.parameter = self.model.parameters()
        self.optimizer = optim.Adam(self.parameter, lr=args.lr / args.batch_size, betas=(0.9, 0.999),
                                    weight_decay=1e-6)

        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, args.lr_scheduler_steps, args.lr_scheduler_gamma)

    def run(self):
        self.train_hist = {
            'loss': [],
            'loss_per_point': [],
            'per_epoch_time': [],
            'total_time': [],
            'eval_time': [],
            'eval_loss': []
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

            # eval model
            if (epoch + 1) % self.eval_interval == 0:
                self._eval(epoch + 1)

            # save tensorboard
            if self.writer:
                self.writer.add_scalar('Train Loss', self.train_hist['loss'][-1], epoch)
                self.writer.add_scalar('Train Loss (per point)', self.train_hist['loss_per_point'][-1], epoch)
                self.writer.add_scalar('Learning Rate', self._get_lr(), epoch)

            self.lr_scheduler.step()

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
        save_dir = os.path.join(self.model_dir, self.dataset_name)
        torch.save(new_state_dict, save_dir + "_" + str(epoch) + '.pkl')
        print(f"Save model to {save_dir}_{str(epoch)}.pkl")
        if self.snapshot_plot:
            self._plot_reconstruction(epoch)

    def _plot_reconstruction(self, epoch, unseen_data=False):
        if unseen_data:
            dir = self.eval_plot_dir
            x = self.dataset.unseen_test
        else:
            dir = self.plot_dir
            x = self.dataset.test_data
        # if not self.no_cuda:
        #     x.cuda(self.first_gpu)
        self.model.eval()
        reconstruction, _ = self.model.forward(x)
        self._plot(x, reconstruction, dir, epoch)
        self.model.train()

    def _plot(self, gt, reconstruction, dir, epoch):
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
        # save plot
        save_dir = os.path.join(dir, self.dataset_name)
        save_dir = save_dir + "_" + str(epoch) + '.png'
        plt.savefig(save_dir)
        print(f"Save model to {save_dir}")

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

    def _eval(self, epoch):
        eval_start_time = time.time()
        self.model.eval()
        pts = self.dataset.get_eval_data()
        if not self.no_cuda:
            pts = pts.cuda(self.first_gpu)

        # forward
        output, _ = self.model(pts)

        # loss
        if len(self.gpu_ids) != 1:  # multiple gpus
            loss = self.model.module.get_loss(pts, output)
        else:
            loss = self.model.get_loss(pts, output)
        loss = loss.detach().cpu().numpy() / self.args.num_points / pts.shape[0]
        self.model.train()

        self._plot_reconstruction(epoch, unseen_data=True)

        # finish one epoch
        eval_time = time.time() - eval_start_time
        self.train_hist['eval_time'].append(eval_time)
        self.train_hist['eval_loss'].append(loss)
        print(f'Eval Epoch {epoch}: Loss {loss}, time {eval_time:.4f}s')
        if self.writer:
            self.writer.add_scalar('Eval Loss', self.train_hist['eval_loss'][-1], epoch)

        if self.eval_plot:
            self._plot_reconstruction(epoch, unseen_data=True)


