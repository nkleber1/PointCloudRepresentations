import torch
import torch.nn as nn
from . import VAEBottleneck, GraphEncoder, GraphDoubleEncoder, GraphEncoderS, PointNet2Encoder, PointNetEncoder, DenseEncoder, FoldDecoder, FoldDecoderS, FoldSingleDecoder, DenseDecoder  # ,PointNet2CudaEncoder
from .loss import ChamferLoss


class ReconstructionNet(nn.Module):
    def __init__(self, args):
        super(ReconstructionNet, self).__init__()
        if args.encoder == 'graph':
            self.encoder = GraphEncoder(args)
        if args.encoder == 'graph_s':
            self.encoder = GraphEncoderS(args)
        if args.encoder == 'graph_double':
            self.encoder = GraphDoubleEncoder(args)
        elif args.encoder == 'pointnet++':
            self.encoder = PointNet2Encoder(args)
        elif args.encoder == 'pointnet2cuda':
            self.encoder = PointNet2CudaEncoder(args)
        elif args.encoder == 'pointnet':
            self.encoder = PointNetEncoder(args)
        elif args.encoder == 'dense':
            self.encoder = DenseEncoder(args)
        if args.decoder == 'fold':
            self.decoder = FoldDecoder(args)
        if args.decoder == 'fold_s':
            self.decoder = FoldDecoderS(args)
        if args.decoder == 'fold_single':
            self.decoder = FoldSingleDecoder(args)
        elif args.decoder == 'upsampling':
            pass
        elif args.decoder == 'dense':
            self.decoder = DenseDecoder(args)
        self.loss = ChamferLoss()
        self.args = args
        self.vae_bottleneck = None
        if not self.args.no_vae:
            self.vae_bottleneck = VAEBottleneck(args)

    def forward(self, input):
        feature, _, _ = self.encode(input)
        output = self.decode(feature)
        return output, feature

    def encode(self, input):
        feature = self.encoder(input)
        mu, std = None, None
        if not self.args.no_vae:
            feature, mu, std = self.vae_bottleneck(feature)
        return feature, mu, std

    def denode(self, feature):
        output = self.decoder(feature)
        return output

    def get_parameter(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def get_loss(self, input, output):
        return self.loss(input, output)

    def load_pretrain(self, pretrain):
        state_dict = torch.load(pretrain, map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key[:6] == 'module':
                name = key[7:]  # remove 'module.'
            else:
                name = key
            new_state_dict[name] = val
        self.load_state_dict(new_state_dict)
        print(f"Load model from {pretrain}")