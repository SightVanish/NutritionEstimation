import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import Nutrition_RGBD  # Assuming Nutrition_RGBD class is saved in a file called Nutrition_RGBD.py

# Set up the transformation for the images
# the input image size is (640, 480)
transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Define the file paths
image_path = 'data/nutrition5k_dataset/imagery'
rgb_txt_path = 'data/nutrition5k_dataset/imagery/rgb_in_overhead_test_processed.txt'
rgbd_txt_path = 'data/nutrition5k_dataset/imagery/rgbd_test_processed.txt'

# Initialize the dataset and data loader
dataset = Nutrition_RGBD(image_path=image_path, rgb_txt_dir=rgb_txt_path, rgbd_txt_dir=rgbd_txt_path, transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Define a naive model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(16 * 640 * 480, 5)  # Adjusting output to 5 for calories, mass, fat, carb, protein

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        return self.fc(x)

# Instantiate the model and set to evaluation mode
model = SimpleModel()
model.eval()

# Load and view the RGBD image
for i, (img_rgb, label, calories, mass, fat, carb, protein, img_rgbd) in enumerate(data_loader):
    print(f"RGB Image shape: {img_rgb.shape}")
    print(f"Depth Image shape: {img_rgbd.shape}")
    
    # Concatenate RGB and Depth images along the channel dimension if needed
    rgbd_input = torch.cat((img_rgb, img_rgbd), dim=1)
    print(f"Concatenated RGBD input shape: {rgbd_input.shape}")
    
    # Forward pass through the model (using only the RGB image for simplicity)
    output = model(img_rgb)
    print(f"Model output: {output}")

    break  # Only run one batch for demonstration