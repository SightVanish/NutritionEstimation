import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights

class RGBDResNet(nn.Module):
    def __init__(self, num_outputs=5):
        super(RGBDResNet, self).__init__()

        # Load pretrained ResNet101 models for RGB and depth images
        self.rgb_resnet = resnet101(weights=ResNet101_Weights.DEFAULT)
        self.depth_resnet = resnet101(weights=ResNet101_Weights.DEFAULT)

        # Modify the first convolutional layer of the depth ResNet if needed
        # Since depth images are 3-channel, we can use the pretrained weights directly

        # Remove the fully connected layers from both models
        self.rgb_features = nn.Sequential(*list(self.rgb_resnet.children())[:-1])  # Exclude the last FC layer
        self.depth_features = nn.Sequential(*list(self.depth_resnet.children())[:-1])

        # Define a fully connected layer to combine features
        self.fc = nn.Sequential(
            nn.Linear(2048 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_outputs)
        )

    def forward(self, rgb_input, depth_input):
        # Extract features from RGB and depth images
        rgb_feat = self.rgb_features(rgb_input)        # Shape: [batch_size, 2048, 1, 1]
        depth_feat = self.depth_features(depth_input)  # Shape: [batch_size, 2048, 1, 1]

        # Flatten the feature maps
        rgb_feat = rgb_feat.view(rgb_feat.size(0), -1)      # Shape: [batch_size, 2048]
        depth_feat = depth_feat.view(depth_feat.size(0), -1)  # Shape: [batch_size, 2048]

        # Concatenate features
        combined_feat = torch.cat((rgb_feat, depth_feat), dim=1)  # Shape: [batch_size, 4096]

        # Pass through the fully connected layers
        output = self.fc(combined_feat)  # Shape: [batch_size, num_outputs]

        return output