import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F

import numpy as np
from necks import BFP

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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


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

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


rgbd = False  # 跑RGBD实验时更改下 lmj 0805-----弃用，因为4通道训练不好。

class ResNet(nn.Module):
    def __init__(self, block, layers, rgbd, bbox, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
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
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # Top layer
        # 用于conv5,因为没有更上一层的特征了，也不需要smooth的部分
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        # 分别用于conv4,conv3,conv2（按顺序）
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        # 分别用于conv4,conv3,conv2（按顺序）
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.rgbd = rgbd
        # 1102
        self.yolobox = bbox
        # 1026
        self.adaAvgPool = nn.AdaptiveAvgPool2d((8, 8))
        # 1102
        self.avgpool_rgbonly = nn.AdaptiveAvgPool2d((1, 1))
        self.fc3 = nn.Linear(1024, 1024)

    # FPN lmj 20210831
    def _upsample_add(self, x, y):
        # 将输入x上采样两倍，并与y相加
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

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
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
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl_bbox(self, x, bbox):
        # Bottom-up  FPN
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # before 1108
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)  # [b, 256, 67, 89]

        output = []
        for i, box in enumerate(bbox):
            if box != '':  # 有几张图片没有bbox
                # pdb.set_trace()
                with open(box, "r+", encoding="utf-8", errors="ignore") as f:
                    # w,h = 89, 67   #resize后的图片
                    w, h = p2.shape[3], p2.shape[2]
                    allLabels = []
                    for line in f:
                        label = []
                        aa = line.split(" ")
                        # pdb.set_trace()
                        x_center = w * float(aa[1])  # aa[1]左上点的x坐标
                        y_center = h * float(aa[2])  # aa[2]左上点的y坐标
                        width = int(w * float(aa[3]))  # aa[3]图片width
                        height = int(h * float(aa[4]))  # aa[4]图片height
                        lefttopx = int(x_center - width / 2.0)
                        lefttopy = int(y_center - height / 2.0)
                        label = [lefttopx, lefttopy, lefttopx + width, lefttopy + height]
                        allLabels.append(label)

                    nparray = np.array(allLabels)
                    # 可能存在多个位置labels
                    lefttopx = nparray[:, 0].min()
                    lefttopy = nparray[:, 1].min()
                    # width = nparray[:,2].max()
                    # height = nparray[:,3].max()
                    left_plus_width = nparray[:, 2].max()
                    top_plus_height = nparray[:, 3].max()

                    # pdb.set_trace()
                    roi = p2[i][..., lefttopy + 1:top_plus_height + 3, lefttopx + 1:left_plus_width + 1]
                    # 池化统一大小
                    output.append(F.adaptive_avg_pool2d(roi, (2, 2)))
            elif box == '':
                # pdb.set_trace()
                output.append(F.adaptive_avg_pool2d(p2[i], (2, 2)))
        output = torch.stack(output, axis=0)
        x = torch.flatten(output, 1)
        x = self.fc3(x)
        x = F.relu(x)
        results = []
        results.append(self.calorie(x).squeeze())  # 2048
        results.append(self.mass(x).squeeze())
        results.append(self.fat(x).squeeze())
        results.append(self.carb(x).squeeze())
        results.append(self.protein(x).squeeze())
        return results

    # Normal
    def _forward_impl(self, x):
        # See note [TorchScript super()]
        if not self.rgbd:
            x = self.conv1(x)  # torch.Size([16, 3, 267, 356])
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)  # torch.Size([16, 2048, 9, 12])

            # pdb.set_trace()
            x = self.avgpool(x)  # 统一进行自适应平均池化，即使输入图片大小不同，x的输出也相同
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = self.fc2(x)
            x = F.relu(x)
            results = []
            results.append(self.calorie(x).squeeze())  # 2048
            results.append(self.mass(x).squeeze())
            results.append(self.fat(x).squeeze())
            results.append(self.carb(x).squeeze())
            results.append(self.protein(x).squeeze())
            return results

        elif self.rgbd:
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
        if self.yolobox:
            return self._forward_impl_bbox(x, bbox)
        else:
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
        cat1 = self.smooth1(cat1)  # torch.Size([16, 512, 32, 32])
        cat2 = self.smooth1(cat2)  # torch.Size([16, 512, 16, 16])
        cat3 = self.smooth1(cat3)  # torch.Size([16, 512, 8, 8])
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

        cat_input = torch.stack([cat0, cat1, cat2, cat3], axis=1)  # torch.Size([16, 4, 512, 1, 1])
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


def resnet101(pretrained=False, progress=True, rgbd=False, bbox=False, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, rgbd, bbox,
                   **kwargs)


def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


#####################################################################################################

if __name__ == '__main__':
    from PIL import Image
    from torchvision import transforms

    model = resnet101(rgbd=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    img = Image.open("/icislab/volume1/swj/nutrition/nutrition5k/nutrition5k_dataset/imagery/realsense_overhead/dish_1556575014/rgb.png")
    transform = transforms.ToTensor()
    img_tensor = transform(img)
    input = torch.randn(1, 3, 256, 256)
    input = input.to(device)
    model.to(device)
    model_cat = Resnet101_concat()
    model_cat.to(device)
    pretrained_dict = torch.load("/icislab/volume1/swj/nutrition/CHECKPOINTS/food2k_resnet101_0.0001.pth")
    now_state_dict = model.state_dict()
    now_state_dict.update(pretrained_dict)
    missing_keys, unexpected_keys = model.load_state_dict(now_state_dict, strict=False)
    out = model(input)
    out_d = model(input)
    results = model_cat(out, out_d)
    print('debug___________________')