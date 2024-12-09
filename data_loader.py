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
                rgb_rel_path = f"{dish_id}/rgb.png"
                depth_color_rel_path = f"{dish_id}/depth_color.png"

                # Nutritional values
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


def split_train_val(dataset: Nutrition5kDataset, val_ratio: float = 0.2) -> Tuple[Dataset, Dataset]:
    """
    Split the training dataset into training and validation sets.

    Args:
        dataset (Nutrition5kDataset): The dataset to split.
        val_ratio (float): Proportion of the dataset to use for validation.

    Returns:
        Tuple[Dataset, Dataset]: Training and validation datasets.
    """
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    return random_split(dataset, [train_size, val_size])


def get_train_val_loaders(root_dir: str, label_file: str, batch_size: int, val_ratio: float = 0.2,
                          num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare and split the Nutrition5k dataset into training and validation DataLoaders.

    Args:
        root_dir (str): Path to the dataset directory.
        label_file (str): Path to the train label file.
        batch_size (int): Number of samples per batch.
        val_ratio (float): Proportion of the training dataset to use for validation.
        num_workers (int): Number of worker threads for data loading.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation DataLoaders.
    """
    # Training dataset and split
    dataset = Nutrition5kDataset(root_dir=root_dir, label_file=label_file, transform=get_transforms(training=True))
    train_dataset, val_dataset = split_train_val(dataset, val_ratio=val_ratio)

    # Update transforms for validation dataset
    val_dataset.dataset.transform = get_transforms(training=False)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


# Example Usage
if __name__ == "__main__":
    root_dir = "data/nutrition5k_dataset/imagery/train"
    label_file = "train.txt"
    batch_size = 32

    # Get DataLoaders
    train_loader, val_loader = get_train_val_loaders(root_dir, label_file, batch_size)

    # Example iteration
    for rgb_images, depth_images, labels in train_loader:
        print(f"RGB batch shape: {rgb_images.size()}")
        print(f"Depth batch shape: {depth_images.size()}")
        print(f"Labels batch shape: {labels.size()}")
        break