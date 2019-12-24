from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torch.utils.model_zoo as model_zoo


__all__ = ['resnet50_SNR']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelGate_sub(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(self, in_channels, num_gates=None, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False):
        super(ChannelGate_sub, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1, bias=True, padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels//reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels//reduction, num_gates, kernel_size=1, bias=True, padding=0)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError("Unknown gate activation: {}".format(gate_activation))

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x, input * (1 - x), x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class UpBlock(nn.Module):
    def __init__(self, inplanes, planes, upsample=False):
        super(UpBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.will_ups = upsample

    def forward(self, x):
        if self.will_ups:
            x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


# class GetParams(nn.Linear):
#     def __init__(self):
#         super(GetParams, self).__init__()
#         for p in self.parameters():
#             p.requires_grad = False
#
#     def forward(self, x):
#         print(self.weight)
#         print(self.bias)

class Conv1x1nonLinear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1nonLinear, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(self.bn(x))
        return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class ResNet_SNR(nn.Module):
    """
    Residual network, SE, 1 - attention

    Reference:
    He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
    """

    def __init__(self, block, layers,
                 last_stride=2,
                 fc_dims=None,
                 dropout_p=None,
                 **kwargs):
        self.inplanes = 64
        super(ResNet_SNR, self).__init__()
        self.feature_dim = 512 * block.expansion

        global args

        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        # for inner triplet loss:
        self.global_avgpool_1 = nn.AdaptiveAvgPool2d(1)
        self.global_avgpool_2 = nn.AdaptiveAvgPool2d(1)
        self.global_avgpool_3 = nn.AdaptiveAvgPool2d(1)
        self.global_avgpool_4 = nn.AdaptiveAvgPool2d(1)

        # IN bridge:
        self.IN1 = nn.InstanceNorm2d(256, affine=True)
        self.IN2 = nn.InstanceNorm2d(512, affine=True)
        self.IN3 = nn.InstanceNorm2d(1024, affine=True)
        self.IN4 = nn.InstanceNorm2d(2048, affine=True)

        # 1*1 Conv for style decomposition:
        self.style_reid_laye1 = ChannelGate_sub(256, num_gates=256, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False)

        self.style_reid_laye2 = ChannelGate_sub(512, num_gates=512, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False)

        self.style_reid_laye3 = ChannelGate_sub(1024, num_gates=1024, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False)

        self.style_reid_laye4 = ChannelGate_sub(2048, num_gates=2048, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False)

        self._init_params()

        # BNN Neck:
        self.BN = nn.BatchNorm1d(512 * block.expansion)
        self.BN.bias.requires_grad_(False)  # no shift
        #self.classifier = nn.Linear(self.feature_dim, num_classes, bias=False)

        self.BN.apply(weights_init_kaiming)
        #self.classifier.apply(weights_init_classifier)

        self.relu_final = nn.ReLU(inplace=False)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """
        Construct fully connected layer

        - fc_dims (list or tuple): dimensions of fc layers, if None,
                                   no fc layers are constructed
        - input_dim (int): input dimension
        - dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(fc_dims, (list, tuple)), "fc_dims must be either list or tuple, but got {}".format(
            type(fc_dims))

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_1 = self.layer1(x)  # torch.Size([64, 256, 64, 32])
        x_1_ori = x_1
        x_IN_1 = self.IN1(x_1)
        x_style_1 = x_1 - x_IN_1
        x_style_1_reid_useful, x_style_1_reid_useless, selective_weight_useful_1 = self.style_reid_laye1(x_style_1)
        x_1 = x_IN_1 + x_style_1_reid_useful
        x_1_useless = x_IN_1 + x_style_1_reid_useless


        x_2 = self.layer2(x_1)  # torch.Size([64, 512, 32, 16])
        x_2_ori = x_2
        x_IN_2 = self.IN2(x_2)
        x_style_2 = x_2 - x_IN_2
        x_style_2_reid_useful, x_style_2_reid_useless, selective_weight_useful_2 = self.style_reid_laye2(x_style_2)
        x_2 = x_IN_2 + x_style_2_reid_useful
        x_2_useless = x_IN_2 + x_style_2_reid_useless

        x_3 = self.layer3(x_2)  # torch.Size([64, 1024, 16, 8])
        x_3_ori = x_3
        x_IN_3 = self.IN3(x_3)
        x_style_3 = x_3 - x_IN_3
        x_style_3_reid_useful, x_style_3_reid_useless, selective_weight_useful_3 = self.style_reid_laye3(x_style_3)
        x_3 = x_IN_3 + x_style_3_reid_useful
        x_3_useless = x_IN_3 + x_style_3_reid_useless

        x_4 = self.layer4(x_3)  # torch.Size([64, 2048, 16, 8])
        x_4_ori = x_4
        x_IN_4 = self.IN4(x_4)
        x_style_4 = x_4 - x_IN_4
        x_style_4_reid_useful, x_style_4_reid_useless, selective_weight_useful_4 = self.style_reid_laye4(x_style_4)
        x_4 = x_IN_4 + x_style_4_reid_useful
        x_4_useless = x_IN_4 + x_style_4_reid_useless

        return x_4, x_3, x_2, x_1, \
               x_4_useless, x_3_useless, x_2_useless, x_1_useless, \
               x_IN_4, x_IN_3, x_IN_2, x_IN_1, \
               x_4_ori, x_3_ori, x_2_ori, x_1_ori, \
               x_style_4_reid_useful, x_style_3_reid_useful, x_style_2_reid_useful, x_style_1_reid_useful, \
               x_style_4_reid_useless, x_style_3_reid_useless, x_style_2_reid_useless, x_style_1_reid_useless, \
               selective_weight_useful_4, selective_weight_useful_3, selective_weight_useful_2, selective_weight_useful_1

    def forward(self, x):
        f_4, f_3, f_2, f_1, \
        f_4_reid_useless, f_3_reid_useless, f_2_reid_useless, f_1_reid_useless, \
        f_IN_4, f_IN_3, f_IN_2, f_IN_1, \
        f_4_ori, f_3_ori, f_2_ori, f_1_ori, \
        f_style_4_reid_useful, f_style_3_reid_useful, f_style_2_reid_useful, f_style_1_reid_useful, \
        f_style_4_reid_useless, f_style_3_reid_useless, f_style_2_reid_useless, f_style_1_reid_useless, \
        selective_weight_useful_4, selective_weight_useful_3, selective_weight_useful_2, selective_weight_useful_1 = self.featuremaps(x)

        v = self.global_avgpool(f_4)
        v = v.view(v.size(0), -1)

        v3 = self.global_avgpool(f_3)
        v3 = v3.view(v3.size(0), -1)
        v2 = self.global_avgpool(f_2)
        v2 = v2.view(v2.size(0), -1)
        v1 = self.global_avgpool(f_1)
        v1 = v1.view(v1.size(0), -1)

        v_bn = self.BN(v)

        v_relu = self.relu_final(v_bn)

        return v_bn, v_relu#torch.cat((v_bn, v3, v2, v1), 1)


def init_pretrained_weights(model, model_url):
    """
    Initialize model with pretrained weights.
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    print("Initialized model with pretrained weights from {}".format(model_url))


"""
Residual network configurations:
--
resnet18: block=BasicBlock, layers=[2, 2, 2, 2]
resnet34: block=BasicBlock, layers=[3, 4, 6, 3]
resnet50: block=Bottleneck, layers=[3, 4, 6, 3]
resnet101: block=Bottleneck, layers=[3, 4, 23, 3]
resnet152: block=Bottleneck, layers=[3, 8, 36, 3]
"""



def resnet50_SNR(**kwargs):
    model = ResNet_SNR(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=[512],
        dropout_p=None,
        **kwargs
    )
    return model