"""
Ref:
https://github.com/hysts/pytorch_wrn/blob/master/wrn.py
https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
https://github.com/szagoruyko/functional-zoo/blob/master/wide-resnet-50-2-export.ipynb
https://github.com/szagoruyko/wide-residual-networks/tree/master/pytorch

to get the name of the layer in state_dict, see the following example
>>> wrn = WideResNet(
>>>         in_channels=1,
>>>         out_channels=5,
>>>         base_channels=4,
>>>         widening_factor=10,
>>>         drop_rate=0,
>>>         depth=10
>>>     )
>>>
>>>     # print the state_dict keys
>>>     d = wrn.state_dict()
>>>     dl = list(d.keys())
>>>     for idx, n in enumerate(dl):
>>>         print("{} -> {}".format(idx, n))
"""
import torch
from torch import nn
import numpy as np

# for type hint
from typing import Union, Tuple, Sequence, Optional
from torch import Tensor


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 drop_rate: float = 0.0,
                 activate_before_residual: bool = False,
                 batch_norm_momentum: float = 0.001):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.drop_rate = drop_rate
        self.equal_in_out = (in_channels == out_channels)
        self.activate_before_residual = activate_before_residual

        # see https://github.com/pytorch/examples/issues/289 regarding different convention for batch norm momentum
        self.bn1 = nn.BatchNorm2d(self.in_channels, momentum=batch_norm_momentum)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)

        self.bn2 = nn.BatchNorm2d(self.out_channels, momentum=batch_norm_momentum)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)

        if not self.equal_in_out:
            self.conv_shortcut = nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False)
        else:
            self.conv_shortcut = None

        if drop_rate > 0:
            self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, inputs):
        if not self.equal_in_out and self.activate_before_residual:
            inputs = self.bn1(inputs)
            inputs = self.relu1(inputs)
            outputs = self.conv1(inputs)
        else:
            outputs = self.bn1(inputs)
            outputs = self.relu1(outputs)
            outputs = self.conv1(outputs)

        outputs = self.bn2(outputs)
        outputs = self.relu2(outputs)
        outputs = self.conv2(outputs)

        if self.drop_rate > 0:
            outputs = self.dropout(outputs)

        if not self.equal_in_out:
            inputs = self.conv_shortcut(inputs)

        return torch.add(inputs, outputs)


class NetworkBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int,
                 stride: int,
                 drop_rate: float = 0.0,
                 activate_before_residual: bool = False,
                 basic_block: type(BasicBlock) = BasicBlock,
                 **block_kwargs):
        super().__init__()
        self.layer = self._make_layer(basic_block, in_channels, out_channels, num_blocks, stride, drop_rate,
                                      activate_before_residual, **block_kwargs)

    @staticmethod
    def _make_layer(block: type(BasicBlock),
                    in_channels: int,
                    out_channels: int,
                    num_blocks: int,
                    stride: int,
                    drop_rate: float,
                    activate_before_residual: bool,
                    **block_kwargs) -> nn.Module:
        layers = []
        for i in range(int(num_blocks)):
            if i == 0:
                layers.append(
                    block(in_channels,
                          out_channels,
                          stride=stride,
                          drop_rate=drop_rate,
                          activate_before_residual=activate_before_residual,
                          **block_kwargs))
            else:
                layers.append(
                    block(out_channels,
                          out_channels,
                          stride=1,
                          drop_rate=drop_rate,
                          activate_before_residual=activate_before_residual,
                          **block_kwargs))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 depth: int = 28,
                 widening_factor: int = 2,
                 base_channels: int = 16,
                 drop_rate: float = 0.0,
                 batch_norm_momentum: float = 0.001,
                 n_channels: Optional[Sequence[int]] = None):
        super().__init__()

        if n_channels is None:
            # interpolate channels
            n_channels = [
                int(base_channels),
                int(base_channels * widening_factor),
                int(base_channels * 2 * widening_factor),
                int(base_channels * 4 * widening_factor),
            ]
        assert len(n_channels) == 4
        self.n_channels = n_channels

        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        self.num_blocks = (depth - 4) // 6

        if self.num_blocks == 0:
            self.fc_in_features = self.n_channels[0]
        else:
            self.fc_in_features = self.n_channels[3]

        self.conv = nn.Conv2d(
            in_channels,
            self.n_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)

        self.block1 = NetworkBlock(
            in_channels=self.n_channels[0],
            out_channels=self.n_channels[1],
            num_blocks=self.num_blocks,
            stride=1,
            drop_rate=drop_rate,
            activate_before_residual=True,
            basic_block=BasicBlock,
            batch_norm_momentum=batch_norm_momentum)

        self.block2 = NetworkBlock(
            in_channels=self.n_channels[1],
            out_channels=self.n_channels[2],
            num_blocks=self.num_blocks,
            stride=2,
            drop_rate=drop_rate,
            basic_block=BasicBlock,
            batch_norm_momentum=batch_norm_momentum)

        self.block3 = NetworkBlock(
            in_channels=self.n_channels[2],
            out_channels=self.n_channels[3],
            num_blocks=self.num_blocks,
            stride=2,
            drop_rate=drop_rate,
            basic_block=BasicBlock,
            batch_norm_momentum=batch_norm_momentum)

        self.bn = nn.BatchNorm2d(num_features=self.fc_in_features, momentum=batch_norm_momentum)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.pool = nn.AvgPool2d(kernel_size=8)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(self.fc_in_features, out_channels)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, inputs: Tensor, return_feature: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        outputs = self.conv(inputs)
        outputs = self.block1(outputs)
        outputs = self.block2(outputs)
        outputs = self.block3(outputs)
        outputs = self.bn(outputs)
        outputs = self.relu(outputs)
        outputs = self.pool(outputs)
        # outputs = outputs.view(-1, self.n_channels[3])
        features = outputs.view(outputs.size(0), -1)

        outputs = self.fc(features)

        if return_feature:
            return features, outputs

        return outputs
