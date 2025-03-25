import torch
import torch.nn as nn

class NutritionCNN(nn.Module):
    def __init__(self):
        super(NutritionCNN, self).__init__()
        
        # RGB image branch
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Depth image branch
        self.depth_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(64 * 32 * 32 * 2, 512),  # 2 branches * 64 channels * 32x32 feature maps
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 5)  # Output: 5 nutritional values
        )
        
    def forward(self, rgb, depth):
        # Process RGB image
        rgb_features = self.rgb_conv(rgb)
        rgb_features = rgb_features.view(rgb_features.size(0), -1)
        
        # Process depth image
        depth_features = self.depth_conv(depth)
        depth_features = depth_features.view(depth_features.size(0), -1)
        
        # Concatenate features
        combined = torch.cat((rgb_features, depth_features), dim=1)
        
        # Regression output
        output = self.fc(combined)
        return output