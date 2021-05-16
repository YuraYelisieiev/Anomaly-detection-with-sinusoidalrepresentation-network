import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmeta.modules import (MetaModule, MetaSequential)
import numpy as np
from collections import OrderedDict
import os
from glob import glob
from torch.utils.data import DataLoader
import dataio
from copy import deepcopy
from collections import namedtuple


class LossNetwork(nn.Module):
    def __init__(self, vgg_model):
        super().__init__()
        self.vgg_layers = vgg_model.features
        self.loss_output = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return self.loss_output(**output)


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(30 * x)


class BatchLinear(nn.Linear, MetaModule):
    """A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.
    Source code: https://github.com/vsitzmann/siren/blob/master/modules.py
    """
    __doc__ = nn.Linear.__doc__

    def forward(self, x, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = x.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output


class FCBlock(MetaModule):
    """A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    Source code: https://github.com/vsitzmann/siren/blob/master/modules.py
    """

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine': (Sine(), sine_init, first_layer_sine_init),
                         'relu': (nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid': (nn.Sigmoid(), init_weights_xavier, None),
                         'tanh': (nn.Tanh(), init_weights_xavier, None),
                         'selu': (nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus': (nn.Softplus(), init_weights_normal, None),
                         'elu': (nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(MetaSequential(
            BatchLinear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(MetaSequential(BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features), nl
            ))

        self.net = MetaSequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None:  # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords, params=self.get_subdict(params, 'net'))
        return output

    def forward_with_activations(self, coords, params=None, retain_grad=False):
        """Returns not only model output, but also intermediate activations."""
        if params is None:
            params = OrderedDict(self.named_parameters())

        activations = OrderedDict()

        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            subdict = self.get_subdict(params, 'net.%d' % i)
            for j, sublayer in enumerate(layer):
                if isinstance(sublayer, BatchLinear):
                    x = sublayer(x, params=self.get_subdict(subdict, '%d' % j))
                else:
                    x = sublayer(x)

                if retain_grad:
                    x.retain_grad()
                activations['_'.join((str(sublayer.__class__), "%d" % i))] = x
        return activations


class SingleBVPNet(MetaModule):
    """A canonical representation network for a BVP."""

    def __init__(self, out_features=1, act_type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()
        self.mode = mode

        if self.mode == 'rbf':
            self.rbf_layer = RBFLayer(in_features=in_features, out_features=kwargs.get('rbf_centers', 1024))
            in_features = kwargs.get('rbf_centers', 1024)
        elif self.mode == 'nerf':
            self.positional_encoding = PosEncodingNeRF(in_features=in_features,
                                                       sidelength=kwargs.get('sidelength', None),
                                                       fn_samples=kwargs.get('fn_samples', None),
                                                       use_nyquist=kwargs.get('use_nyquist', True))
            in_features = self.positional_encoding.out_dim

        self.image_downsampling = ImageDownsampling(sidelength=kwargs.get('sidelength', None),
                                                    downsample=kwargs.get('downsample', False))
        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=act_type)
        print(self)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        coords = coords_org

        # various input processing methods for different applications
        if self.image_downsampling.downsample:
            coords = self.image_downsampling(coords)
        if self.mode == 'rbf':
            coords = self.rbf_layer(coords)
        elif self.mode == 'nerf':
            coords = self.positional_encoding(coords)

        output = self.net(coords, self.get_subdict(params, 'net'))
        return {'model_in': coords_org, 'model_out': output}

    def forward_with_activations(self, model_input):
        """Returns not only model output, but also intermediate activations."""
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        activations = self.net.forward_with_activations(coords)
        return {'model_in': coords, 'model_out': activations.popitem(), 'activations': activations}


class ImageDownsampling(nn.Module):
    """Generate samples in u,v plane according to downsampling blur kernel"""

    def __init__(self, sidelength, downsample=False):
        super().__init__()
        if isinstance(sidelength, int):
            self.sidelength = (sidelength, sidelength)
        else:
            self.sidelength = sidelength

        if self.sidelength is not None:
            self.sidelength = torch.Tensor(self.sidelength).cuda().float()
        else:
            assert downsample is False
        self.downsample = downsample

    def forward(self, coords):
        if self.downsample:
            return coords + self.forward_bilinear(coords)
        else:
            return coords

    def forward_box(self, coords):
        return 2 * (torch.rand_like(coords) - 0.5) / self.sidelength

    def forward_bilinear(self, coords):
        Y = torch.sqrt(torch.rand_like(coords)) - 1
        Z = 1 - torch.sqrt(torch.rand_like(coords))
        b = torch.rand_like(coords) < 0.5

        Q = (b * Y + ~b * Z) / self.sidelength
        return Q


class PosEncodingNeRF(nn.Module):
    """Module to add positional encoding as in NeRF [Mildenhall et al. 2020]."""

    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        elif self.in_features == 1:
            assert fn_samples is not None
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)


class CropConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, overlap, center_size,
                 bias=False, first=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.overlap = overlap
        self.center_size = center_size
        self.first = first

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        batch_size, in_channels, in_h, in_w = x.shape

        crop_start = (in_h - in_h // self.center_size) // 2 - self.overlap
        start_idx = in_h * crop_start + crop_start
        crop_size = (in_h // self.center_size) + self.overlap * 2
        not_cropped = crop_start * 2

        out_h = ((in_h - self.kernel_size + 2 * self.padding) // self.stride + 1)
        out_w = ((in_w - self.kernel_size + 2 * self.padding) // self.stride + 1)

        unfold = torch.nn.Unfold(kernel_size=(self.kernel_size, self.kernel_size), dilation=self.dilation,
                                 padding=self.padding, stride=self.stride)
        inp_unf = unfold(x)
        crop_lst = []
        for i in range(crop_size):
            if i == 0:
                crop_lst.append(
                    torch.ones([batch_size, self.out_channels, inp_unf[:, :, :start_idx].shape[2]], dtype=torch.bool))
            if i == crop_size - 1:
                crop_lst.append(torch.zeros(
                    [batch_size, self.out_channels, inp_unf[:, :, start_idx:start_idx + crop_size].shape[2]],
                    dtype=torch.bool))
                crop_lst.append(torch.ones(
                    [batch_size, self.out_channels, inp_unf[:, :, start_idx + crop_size:(in_h ** 2)].shape[2]],
                    dtype=torch.bool))
                break

            crop_lst.append(
                torch.zeros([batch_size, self.out_channels, inp_unf[:, :, start_idx:start_idx + crop_size].shape[2]],
                            dtype=torch.bool))
            crop_lst.append(torch.ones(
                [batch_size, self.out_channels, inp_unf[:, :, start_idx + crop_size: start_idx + not_cropped].shape[2]],
                dtype=torch.bool))
            start_idx += in_h

        crop_indexes = torch.cat(crop_lst, axis=2).cuda()

        out_unf = inp_unf.transpose(1, 2).matmul(self.weight.view(self.weight.size(0), -1).t())

        if self.bias is None:
            out_unf = out_unf.transpose(1, 2)
        else:
            out_unf = (out_unf + self.bias).transpose(1, 2)
        out_unf = torch.where(crop_indexes, out_unf, torch.zeros(out_unf.shape, dtype=torch.float32).cuda())
        out = out_unf.view(batch_size, self.out_channels, out_h, out_w)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            CropConv(in_channels=1, out_channels=256, kernel_size=3, stride=2, padding=1, dilation=1, overlap=0,
                     center_size=4, first=True),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            CropConv(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, dilation=1, center_size=4,
                     overlap=0),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            CropConv(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, dilation=1, center_size=4,
                     overlap=0),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear = nn.Sequential(
            nn.Linear(4096, 512),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size()[0], -1)

        encoded = self.linear(encoded)
        return encoded


class CropConvNeuralProcess(nn.Module):
    def __init__(self, in_features, image_resolution, hidden_layers=5, hidden_features=1024):
        super().__init__()
        self.encoder = Encoder()
        self.image_resolution = image_resolution
        self.siren = SingleBVPNet(act_type="sine", mode='mlp', sidelength=self.image_resolution,
                                  in_features=in_features,
                                  num_hidden_layers=hidden_layers, hidden_features=hidden_features)

    def forward(self, model_input):
        batch_size, feature_number, _ = model_input['coords'].shape
        borders = self.encoder(model_input['border']).unsqueeze(1)

        borders = borders.repeat(1, feature_number, 1)
        model_input['coords'] = torch.cat((model_input['coords'], borders), 2)
        model_output = self.siren(model_input)

        return model_output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5),
                                     nn.PReLU(),
                                     nn.Dropout2d(p=0.2),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5),
                                     nn.PReLU(),
                                     nn.Dropout2d(p=0.2),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(64, 128, 5),
                                     nn.PReLU(),
                                     nn.Dropout2d(p=0.2),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(128 * 4 * 4, 512))

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output


class TripletNet(nn.Module):
    def __init__(self, emb_net):
        super(TripletNet, self).__init__()
        self.embedding_net = emb_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # grab from upstream pytorch branch and paste here for now
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
