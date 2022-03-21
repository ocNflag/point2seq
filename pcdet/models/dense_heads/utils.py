import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys
from collections import OrderedDict
from pcdet.ops.dcn import DeformConv

SIGMOID_THRESH = 1e-4


def _sigmoid(x):
    rlt = torch.clamp(x.sigmoid(), min=SIGMOID_THRESH, max=1 - SIGMOID_THRESH)
    return rlt


def kaiming_init(module, a=0, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class Sequential(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input


class FeatureAdaption(nn.Module):
    """Feature Adaption Module.

    Feature Adaption Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deformable conv layer.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Deformable conv kernel size.
        deformable_groups (int): Deformable conv group size.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=4):
        super(FeatureAdaption, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(in_channels, deformable_groups * offset_channels, 1, bias=True)
        self.conv_adaption = DeformConv(in_channels, out_channels, kernel_size=kernel_size,
                                        padding=(kernel_size - 1) // 2, deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def init_weights(self):
        pass

    def forward(self, x, ):
        offset = self.conv_offset(x)
        x = self.relu(self.conv_adaption(x, offset))
        return x


class NaiveSepHead(nn.Module):
    def __init__(
            self,
            in_ch,
            heads,
            head_ch=64,
            name="",
            final_kernel=1,
            bn=False,
            init_bias=-2.19,
            directional_classifier=False,
            **kwargs,
    ):
        super(NaiveSepHead, self).__init__(**kwargs)

        self.heads = heads
        for head in self.heads:
            classes, num_conv = self.heads[head]

            fc = Sequential()
            for i in range(num_conv - 1):
                fc.add(nn.Conv2d(in_ch, head_ch,
                                 kernel_size=final_kernel, stride=1,
                                 padding=final_kernel // 2, bias=True))
                if bn:
                    fc.add(nn.BatchNorm2d(head_ch))
                fc.add(nn.ReLU())

            fc.add(nn.Conv2d(head_ch, classes,
                             kernel_size=final_kernel, stride=1,
                             padding=final_kernel // 2, bias=True))

            if ('center_map' in head) or ('corner_map' in head) or ('foreground_map' in head):
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)

            self.__setattr__(head, fc)

        assert directional_classifier is False, "Doesn't work well with nuScenes in my experiments, please open a pull request if you are able to get it work. We really appreciate contribution for this."

    def forward(self, x):
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict
