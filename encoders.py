from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import BasicBlock,Bottleneck,conv1x1
from torchvision.utils import _log_api_usage_once
from torch.nn import functional as F

class ResNet_enc(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def resnet18_encode():

    return ResNet_enc(block = BasicBlock, layers=[2, 2, 2, 2])

def resnet34_encode():
    return ResNet_enc(block = BasicBlock, layers=[3, 4, 6, 3])

def resnet50_encode():

    return ResNet_enc(block = Bottleneck, layers=[3, 4, 6, 3])

class enc_classifier(nn.Module):

    def __init__(self, enc_path,resnet='resnet18', num_classes = 2):
        super().__init__()
        if resnet == 'resnet18':
            self.encoder = resnet18_encode()
        if resnet == 'resnet34':
            self.encoder = resnet34_encode()
        if resnet == 'resnet50':
            self.encoder = resnet50_encode()
        self.encoder.load_state_dict(torch.load(enc_path,weights_only=True)['model_state_dict'])
        #We tryied leaving more layers frozen but the full fine tune worked best
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        # for param in self.encoder.layer4.parameters():
        #     param.requires_grad = True
        in_features = self.encoder.fc.in_features
        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)  # New output layer
        )
    def forward(self,x):
        x = self.encoder(x)
        return self.fc(x)


class Net(nn.Module):
    """
    Simple CNN also used for the straight training comparison in the report.
    We'll be using the parameters:

    channels_conv0 =  3
    channels_conv1 = 16
    channels_conv2 = 32

    kernel_size_conv1 = 5
    kernel_size_conv2 = 5

    num_features0 = 224
    num_features0 = (num_features0 - kernel_size_conv1 + 1) // 2
    num_features0 = (num_features0 - kernel_size_conv2 + 1) // 2
    num_features0 = num_features0 * num_features0 * channels_conv2

    num_features1 = 110
    num_features2 = 84
    """
    def __init__(self) -> None:
        super(Net, self).__init__()

        channels_conv0 =  3
        channels_conv1 = 16
        channels_conv2 = 32

        kernel_size_conv1 = 5
        kernel_size_conv2 = 5

        num_features0 = 224
        num_features0 = (num_features0 - kernel_size_conv1 + 1) // 2
        num_features0 = (num_features0 - kernel_size_conv2 + 1) // 2
        num_features0 = num_features0 * num_features0 * channels_conv2

        num_features1 = 110
        num_features2 = 53

        self.conv1 = nn.Conv2d(channels_conv0, channels_conv1, kernel_size_conv1)
        self.conv2 = nn.Conv2d(channels_conv1, channels_conv2, kernel_size_conv2)

        self.fc1 = nn.Linear(num_features0, num_features1)
        self.fc2 = nn.Linear(num_features1, num_features2)
        self.fc3 = nn.Linear(num_features2, 2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        X = input / 255.0

        # Convolutional block 1
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, (2, 2))

        # Convolutional block 2
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, (2, 2))

        # Dense block
        X = torch.flatten(X, 1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        output = self.fc3(X)

        return output