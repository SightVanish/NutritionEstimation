import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F

from necks.bfp import BFP

__all__ = ['ResNet', 'resnet101']

model_urls = {
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        layer1 = self.relu(bn1)

        conv2 = self.conv2(layer1)
        bn2 = self.bn2(conv2)
        layer2 = self.relu(bn2)

        conv3 = self.conv3(layer2)
        bn3 = self.bn3(conv3)

        if self.downsample is not None:
            identity = self.downsample(x)

        bn3 += identity
        out = self.relu(bn3)

        return out


class ResNet(nn.Module):
    # block = Bottleneck
    # layers = [3, 4, 23, 3]
    # rgb-d = true
    # bbox = False
    def __init__(self, block, layers, rgbd, bbox, num_classes=1000,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet, self).__init__()

        self._norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0) # c4
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0) # c3
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0) # c2

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # Bottom-up  FPN
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)  # torch.Size([1, 2048, 8, 8]) when image input ==(256,256)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2, p3, p4, p5

    def forward(self, x, bbox=None):
        return self._forward_impl(x)

class Resnet101_concat(nn.Module):
    def __init__(self):
        super(Resnet101_concat, self).__init__()
        self.refine = BFP(512, 4)

        self.smooth1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.smooth4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.ca0 = ChannelAttention(512)
        self.sa0 = SpatialAttention()
        self.ca1 = ChannelAttention(512)
        self.sa1 = SpatialAttention()
        self.ca2 = ChannelAttention(512)
        self.sa2 = SpatialAttention()
        self.ca3 = ChannelAttention(512)
        self.sa3 = SpatialAttention()

        self.avgpool_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_2 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_3 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_4 = nn.AdaptiveAvgPool2d((1, 1))

        self.calorie = nn.Sequential(nn.Linear(1024, 1024), nn.Linear(1024, 1))
        self.mass = nn.Sequential(nn.Linear(1024, 1024), nn.Linear(1024, 1))
        self.fat = nn.Sequential(nn.Linear(1024, 1024), nn.Linear(1024, 1))
        self.carb = nn.Sequential(nn.Linear(1024, 1024), nn.Linear(1024, 1))
        self.protein = nn.Sequential(nn.Linear(1024, 1024), nn.Linear(1024, 1))
        self.fc = nn.Linear(2048, 1024)
        self.LayerNorm = nn.LayerNorm(2048)

    # 4向量融合，一个result
    def forward(self, rgb, rgbd):
        cat0 = torch.cat((rgb[0], rgbd[0]), 1)  # torch.Size([16, 512, 64, 64])
        cat1 = torch.cat((rgb[1], rgbd[1]), 1)  # torch.Size([16, 512, 32, 32])
        cat2 = torch.cat((rgb[2], rgbd[2]), 1)  # torch.Size([16, 512, 16, 16])
        cat3 = torch.cat((rgb[3], rgbd[3]), 1)  # torch.Size([16, 512, 8, 8])
        # BFP
        cat0, cat1, cat2, cat3 = self.refine(tuple((cat0, cat1, cat2, cat3)))
        cat0 = self.smooth1(cat0)  # torch.Size([16, 512, 64, 64])
        cat1 = self.smooth2(cat1)  # torch.Size([16, 512, 32, 32])
        cat2 = self.smooth3(cat2)  # torch.Size([16, 512, 16, 16])
        cat3 = self.smooth4(cat3)  # torch.Size([16, 512, 8, 8])
        # CMBA
        cat0 = self.ca0(cat0) * cat0
        cat0 = self.sa0(cat0) * cat0
        cat1 = self.ca1(cat1) * cat1
        cat1 = self.sa1(cat1) * cat1
        cat2 = self.ca2(cat2) * cat2
        cat2 = self.sa2(cat2) * cat2
        cat3 = self.ca3(cat3) * cat3
        cat3 = self.sa3(cat3) * cat3

        cat0 = self.avgpool_1(cat0)
        cat1 = self.avgpool_2(cat1)
        cat2 = self.avgpool_3(cat2)
        cat3 = self.avgpool_4(cat3)

        cat_input = torch.stack([cat0, cat1, cat2, cat3], dim=1)  # torch.Size([16, 4, 512, 1, 1])
        input = cat_input.view(cat_input.shape[0], -1)  # torch.Size([N, 5, 1024]) N =16(bz) 11(最后batch图片不足)
        input = self.fc(input)
        input = F.relu(input)  # torch.Size([16, 2048]) 添加原因：faster rcnn 也加了

        results = []
        results.append(self.calorie(input).squeeze())
        results.append(self.mass(input).squeeze())
        results.append(self.fat(input).squeeze())
        results.append(self.carb(input).squeeze())
        results.append(self.protein(input).squeeze())

        return results

def _resnet(arch, block, layers, pretrained, progress, rgbd, bbox, **kwargs):
    model = ResNet(block, layers, rgbd, bbox, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet101(pretrained=False, progress=True, rgbd=True, bbox=False, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, rgbd, bbox,
                   **kwargs)
