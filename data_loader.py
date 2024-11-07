import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from typing import Optional

class Nutrition5kDataset(Dataset):
    def __init__(self, root_dir: str, label_file: str, transform: Optional[transforms.Compose] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        label_path = os.path.join(self.root_dir, label_file)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                rgb_rel_path = parts[0]  # Path to rgb.png
                labels = list(map(float, parts[2:]))
                depth_dir = os.path.dirname(rgb_rel_path)
                depth_color_rel_path = os.path.join(depth_dir, 'depth_color.png')
                rgb_path = os.path.join(self.root_dir, rgb_rel_path)
                depth_color_path = os.path.join(self.root_dir, depth_color_rel_path)
                self.samples.append({
                    'rgb_path': rgb_path,
                    'depth_color_path': depth_color_path,
                    'labels': torch.tensor(labels, dtype=torch.float32)
                })
        # Define transforms if not provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Load images
        rgb_image = Image.open(sample['rgb_path']).convert('RGB')
        depth_image = Image.open(sample['depth_color_path']).convert('RGB')
        # Apply transforms
        if self.transform:
            rgb_image = self.transform(rgb_image)
            depth_image = self.transform(depth_image)
        labels = sample['labels']
        return rgb_image, depth_image, labels


# Function to get the dataset
def get_nutrition5k_dataset(root_dir: str, label_file: str) -> Nutrition5kDataset:
    """Returns an instance of the Nutrition5kDataset."""
    return Nutrition5kDataset(root_dir=root_dir, label_file=label_file)

# Function to get the DataLoader
def get_dataloader(dataset: torch.utils.data.Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 4) -> DataLoader:
    """Returns a DataLoader for the given dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# Define a naive model for testing
class NaiveModel(nn.Module):
    def __init__(self):
        super(NaiveModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, stride=2, padding=1),  # Input channels changed to 6
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 14 * 14, 128),  # Adjust input features based on the output size after conv layers
            nn.ReLU(),
            nn.Linear(128, 5)  # Output 5 values corresponding to the labels
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x

# Main testing code
if __name__ == '__main__':
    # Set the root directory and label file paths
    root_dir = 'data/nutrition5k_dataset/imagery'  # Replace with your actual path
    label_file = 'label_train.txt'

    # Get the dataset and dataloader
    dataset = get_nutrition5k_dataset(root_dir, label_file)
    dataloader = get_dataloader(dataset, batch_size=4, shuffle=True)  # Using batch_size=4 for testing

    # Initialize the model, loss function, and optimizer
    model = NaiveModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Move model to GPU if available

    if torch.backends.mps.is_available():
        print("Training with MPS backend.")
    else:
        print("MPS backend is not available.")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    # Testing loop (one epoch)
    model.train()
    for batch_idx, (rgb_image, depth_image, labels) in enumerate(dataloader):
        inputs = torch.cat((rgb_image, depth_image), dim=1)
        
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        print(f'Batch {batch_idx+1}, Loss: {loss.item():.4f}')

        # For testing, we'll just run one batch
        if batch_idx == 0:
            break

    print('Data loading and model testing completed successfully.')