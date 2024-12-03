import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from typing import Optional, Tuple

class Nutrition5kDataset(Dataset):
    def __init__(self, root_dir: str, label_file: str, transform: Optional[transforms.Compose] = None):
        """
        Initialize the Nutrition5kDataset.

        Args:
            root_dir (str): Path to the root directory of the dataset.
            label_file (str): Path to the label file (relative to root_dir).
            transform (Optional[transforms.Compose]): Transformations to apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        label_path = os.path.join(self.root_dir, label_file)
        
        # Parse the label file
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(',')
                
                # Extract paths and label values
                dish_id = parts[0]
                rgb_rel_path = f"imagery/realsense_overhead/{dish_id}/rgb.png"
                depth_color_rel_path = f"imagery/realsense_overhead/{dish_id}/depth_color.png"
                
                # Nutritional values (adjust indices based on the actual structure)
                calories, fat, protein, carbs = map(float, parts[1:5])
                labels = torch.tensor([calories, fat, protein, carbs], dtype=torch.float32)
                
                # Construct absolute paths
                rgb_path = os.path.join(self.root_dir, rgb_rel_path)
                depth_color_path = os.path.join(self.root_dir, depth_color_rel_path)
                
                # Store the sample
                self.samples.append({
                    'rgb_path': rgb_path,
                    'depth_color_path': depth_color_path,
                    'labels': labels
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Transformed RGB image, depth image, and labels.
        """
        sample = self.samples[idx]
        rgb_image = Image.open(sample['rgb_path']).convert('RGB')
        depth_image = Image.open(sample['depth_color_path']).convert('RGB')
        labels = sample['labels']
        if self.transform:
            rgb_image = self.transform(rgb_image)
            depth_image = self.transform(depth_image)
        return rgb_image, depth_image, labels

def get_transforms(training: bool = True) -> transforms.Compose:
    """
    Get data transformations.

    Args:
        training (bool): If True, apply data augmentation.

    Returns:
        transforms.Compose: Data transformations.
    """
    if training:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

def split_dataset(dataset: Nutrition5kDataset, 
                  train_ratio: float = 0.7, 
                  val_ratio: float = 0.15) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split the dataset into training, validation, and testing sets.

    Args:
        dataset (Nutrition5kDataset): The dataset to split.
        train_ratio (float): Proportion of the dataset to use for training.
        val_ratio (float): Proportion of the dataset to use for validation.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: Training, validation, and testing datasets.
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size])

def get_nutrition5k_datasets(root_dir: str, label_file: str) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Prepare and split the Nutrition5k dataset.

    Args:
        root_dir (str): Path to the dataset directory.
        label_file (str): Path to the label file.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: Training, validation, and testing datasets.
    """
    dataset = Nutrition5kDataset(root_dir=root_dir, label_file=label_file)
    train_transform = get_transforms(training=True)
    test_transform = get_transforms(training=False)
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)
    
    # Assign different transforms to datasets
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = test_transform
    test_dataset.dataset.transform = test_transform
    
    return train_dataset, val_dataset, test_dataset

def get_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 4) -> DataLoader:
    """
    Get a DataLoader for the given dataset.

    Args:
        dataset (Dataset): The dataset to load.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of worker threads for data loading.

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# Example Usage
if __name__ == "__main__":
    root_dir = "data/nutrition5k_dataset"
    label_file = "imagery/label.txt"

    # Prepare datasets
    train_dataset, val_dataset, test_dataset = get_nutrition5k_datasets(root_dir, label_file)

    # Create DataLoaders
    train_loader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    # Iterate over a DataLoader
    for rgb_images, depth_images, labels in train_loader:
        print(f"RGB batch shape: {rgb_images.size()}")
        print(f"Depth batch shape: {depth_images.size()}")
        print(f"Labels batch shape: {labels.size()}")
        break