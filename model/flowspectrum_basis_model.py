from __future__ import print_function, division, absolute_import
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo

__all__ = ['FBResNet', 'fbresnet50']

model_urls = {
    'fbresnet50': 'http://data.lip6.fr/cadene/pretrainedmodels/resnet50-19c8e357.pth',
}


class FBResNet(nn.Module):

    def __init__(self, num_segments, block, layers, num_classes=1000):
        self.inplanes = 64

        self.input_space = None
        self.input_size = (224, 224, 3)
        self.mean = None
        self.std = None
        self.num_segments = num_segments
        super(FBResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.num_segments, Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer(self.num_segments, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(self.num_segments, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(self.num_segments, block, 512, layers[3], stride=2)
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, num_segments, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(num_segments, self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(num_segments, self.inplanes, planes))

        return nn.Sequential(*layers)

    def features(self, input):
        x = self.conv1(input)
        self.conv1_input = x.clone()
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, features):
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def fbresnet50(num_segments=8, pretrained=False, num_classes=1000):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FBResNet(num_segments, BottleneckShift, [3, 4, 6, 3], num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['fbresnet50']), strict=False)
    return model


class TransformOperator(nn.Module):
    def __init__(self, channel, n_segment=8, index=1):
        super(TransformOperator, self).__init__()
        self.channel = channel
        self.n_segment = n_segment
        self.reduction = 16
        self.stride = 2 ** (index - 1)

        reduced_channel = self.channel // self.reduction

        self.conv1 = nn.Conv2d(self.channel, reduced_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(reduced_channel)
        self.conv2 = nn.Conv2d(reduced_channel, reduced_channel, kernel_size=3, padding=1, groups=reduced_channel,
                               bias=False)

        self.avg_pool_forward2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool_forward4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.avg_pool_backward2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool_backward4 = nn.AvgPool2d(kernel_size=4, stride=4)

        self.sigmoid = nn.Sigmoid()

        self.pad_forward = (0, 0, 0, 0, 0, 0, 0, 1)
        self.pad_backward = (0, 0, 0, 0, 0, 0, 1, 0)

        self.conv3 = nn.Conv2d(reduced_channel, self.channel, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.channel)

        self.conv3_smallscale2 = nn.Conv2d(reduced_channel, reduced_channel, kernel_size=3, padding=1, bias=False)
        self.bn3_smallscale2 = nn.BatchNorm2d(reduced_channel)

        self.conv3_smallscale4 = nn.Conv2d(reduced_channel, reduced_channel, kernel_size=3, padding=1, bias=False)
        self.bn3_smallscale4 = nn.BatchNorm2d(reduced_channel)

    def forward(self, x):
        bottleneck = self.bn1(self.conv1(x))
        reshaped_bottleneck = bottleneck.view(-1, self.n_segment, *bottleneck.shape[1:])

        t_fea_forward = reshaped_bottleneck[:, :-1]
        t_fea_backward = reshaped_bottleneck[:, 1:]

        conv_bottleneck = self.conv2(bottleneck)
        reshaped_conv_bottleneck = conv_bottleneck.view(-1, self.n_segment, *conv_bottleneck.shape[1:])

        t1_fea_forward = reshaped_conv_bottleneck[:, 1:]
        t1_fea_backward = reshaped_conv_bottleneck[:, :-1]

        diff_fea_forward = t1_fea_forward - t_fea_forward
        diff_fea_backward = t1_fea_backward - t_fea_backward

        diff_fea_0_forward = F.pad(diff_fea_forward, self.pad_forward)
        diff_fea_0_backward = F.pad(diff_fea_backward, self.pad_backward)

        diff_fea_0_forward = diff_fea_0_forward.view(-1, *diff_fea_0_forward.shape[2:])
        diff_fea_0_backward = diff_fea_0_backward.view(-1, *diff_fea_0_backward.shape[2:])

        y_forward_smallscale2 = self.bn3_smallscale2(
            self.conv3_smallscale2(self.avg_pool_forward2(diff_fea_0_forward)))
        y_backward_smallscale2 = self.bn3_smallscale2(
            self.conv3_smallscale2(self.avg_pool_backward2(diff_fea_0_backward)))

        y_forward_smallscale4 = self.bn3_smallscale4(self.conv3_smallscale4(diff_fea_0_forward))
        y_backward_smallscale4 = self.bn3_smallscale4(self.conv3_smallscale4(diff_fea_0_backward))

        y_forward_smallscale2 = F.interpolate(y_forward_smallscale2, diff_fea_0_forward.shape[2:])
        y_backward_smallscale2 = F.interpolate(y_backward_smallscale2, diff_fea_0_backward.shape[2:])

        y_forward = self.bn3(
            self.conv3((diff_fea_0_forward + y_forward_smallscale2 + y_forward_smallscale4) / 3))
        y_backward = self.bn3(
            self.conv3((diff_fea_0_backward + y_backward_smallscale2 + y_backward_smallscale4) / 3))

        y_forward = self.sigmoid(y_forward) - 0.5
        y_backward = self.sigmoid(y_backward) - 0.5

        y = 0.5 * (y_forward + y_backward)
        output = x + x * y
        return output


class ShiftModule(nn.Module):
    def __init__(self, input_channels, n_segment=8, n_div=8, mode='shift'):
        super(ShiftModule, self).__init__()
        self.input_channels = input_channels
        self.n_segment = n_segment
        self.fold_div = n_div
        self.fold = self.input_channels // self.fold_div
        self.conv = nn.Conv1d(self.fold_div * self.fold, self.fold_div * self.fold,
                              kernel_size=3, padding=1, groups=self.fold_div * self.fold,
                              bias=False)

        if mode == 'shift':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:self.fold, 0, 2] = 1  # shift left
            self.conv.weight.data[self.fold: 2 * self.fold, 0, 0] = 1  # shift right
            if 2 * self.fold < self.input_channels:
                self.conv.weight.data[2 * self.fold:, 0, 1] = 1  # fixed
        elif mode == 'fixed':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:, 0, 1] = 1  # fixed
        elif mode == 'norm':
            self.conv.weight.requires_grad = True

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        x = x.permute(0, 3, 4, 2, 1)  # (n_batch, h, w, c, n_segment)
        x = x.contiguous().view(n_batch * h * w, c, self.n_segment)
        x = self.conv(x)  # (n_batch*h*w, c, n_segment)
        x = x.view(n_batch, h, w, c, self.n_segment)
        x = x.permute(0, 4, 3, 1, 2)  # (n_batch, n_segment, c, h, w)
        x = x.contiguous().view(nt, c, h, w)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, num_segments, inplanes, planes, stride=1, downsample=None):
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

    def __init__(self, num_segments, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class BottleneckShift(nn.Module):
    expansion = 4

    def __init__(self, num_segments, inplanes, planes, stride=1, downsample=None):
        super(BottleneckShift, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.num_segments = num_segments
        self.operator = TransformOperator(planes, n_segment=self.num_segments, index=1)
        self.shift = ShiftModule(planes, n_segment=self.num_segments, n_div=8, mode='shift')

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.shift(out)

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
