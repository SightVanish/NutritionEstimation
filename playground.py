import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101, ResNet101_Weights

# Helper Modules
class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        x_channel = self.channel_attention(x) * x
        # Spatial Attention
        avg_out = torch.mean(x_channel, dim=1, keepdim=True)
        max_out, _ = torch.max(x_channel, dim=1, keepdim=True)
        x_spatial = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        return x_channel * x_spatial


class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network (FPN) for multi-scale fusion."""
    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list])
        self.fpn_convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list])

    def forward(self, inputs):
        # Lateral connections
        laterals = [lateral_conv(x) for x, lateral_conv in zip(inputs, self.lateral_convs)]
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode='nearest')
        # FPN output
        outputs = [fpn_conv(lateral) for lateral, fpn_conv in zip(laterals, self.fpn_convs)]
        return outputs


class RGBDFusionNetwork(nn.Module):
    """RGB-D Fusion Network for Food Nutrition Estimation."""
    def __init__(self, backbone=resnet101, num_tasks=5, feature_channels=256):
        super(RGBDFusionNetwork, self).__init__()
        # Load backbone
        self.backbone_rgb = backbone(weights=ResNet101_Weights.DEFAULT)
        self.backbone_depth = backbone(weights=ResNet101_Weights.DEFAULT)

        # Feature extraction layers
        self.rgb_layers = nn.ModuleList([
            self.backbone_rgb.conv1,
            self.backbone_rgb.bn1,
            self.backbone_rgb.relu,
            self.backbone_rgb.maxpool,
            self.backbone_rgb.layer1,
            self.backbone_rgb.layer2,
            self.backbone_rgb.layer3,
            self.backbone_rgb.layer4
        ])

        self.depth_layers = nn.ModuleList([
            self.backbone_depth.conv1,
            self.backbone_depth.bn1,
            self.backbone_depth.relu,
            self.backbone_depth.maxpool,
            self.backbone_depth.layer1,
            self.backbone_depth.layer2,
            self.backbone_depth.layer3,
            self.backbone_depth.layer4
        ])

        # FPN for Multi-Scale Fusion
        self.fpn = FeaturePyramidNetwork([256, 512, 1024, 2048], feature_channels)

        # CBAM for Multimodal Fusion
        self.cbam = CBAM(feature_channels)

        # Fully Connected Layers for Prediction
        self.fc = nn.Linear(feature_channels * 4, num_tasks)

    def extract_features(self, x, layers):
        features = []
        for i, layer in enumerate(layers):
            x = layer(x)
            if i >= 4:  # Collect features from layer1 to layer4
                features.append(x)
        return features

    def forward(self, rgb, depth):
        # Extract features
        rgb_features = self.extract_features(rgb, self.rgb_layers)
        depth_features = self.extract_features(depth, self.depth_layers)

        # Multi-scale fusion with FPN
        rgb_fused = self.fpn(rgb_features)
        depth_fused = self.fpn(depth_features)

        # Element-wise addition
        fused_features = [r + d for r, d in zip(rgb_fused, depth_fused)]

        # Attention-based enhancement
        attention_features = [self.cbam(f) for f in fused_features]

        # Global Average Pooling
        pooled_features = torch.cat([F.adaptive_avg_pool2d(f, 1).flatten(1) for f in attention_features], dim=1)

        # Final Prediction
        output = self.fc(pooled_features)
        return output


# Example Usage
if __name__ == "__main__":
    # Model Initialization
    model = RGBDFusionNetwork()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Dummy Data
    rgb_input = torch.rand((4, 3, 224, 224)).to(device)
    depth_input = torch.rand((4, 3, 224, 224)).to(device)

    # Forward Pass
    predictions = model(rgb_input, depth_input)
    print(predictions)