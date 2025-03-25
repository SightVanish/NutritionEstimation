import os
import random
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class Nutrition5KDataset(Dataset):
    def __init__(self, root_dir, n_samples=1000, transform=None, split='train', train_test_ratio=0.7, random_seed=42):
        """
        Nutrition5K Dataset Loader
        
        Args:
            root_dir (str): Root directory of the Nutrition5K dataset
            n_samples (int): Number of samples to load (default: 1000)
            transform (callable, optional): Optional transform to be applied on a sample
            split (str): 'train' or 'test' to specify which split to load
            train_test_ratio (float): Ratio of training to test data (default: 0.7)
            random_seed (int): Random seed for reproducibility (default: 42)
        """
        self.root_dir = root_dir
        self.n_samples = n_samples
        self.transform = transform
        self.split = split
        self.train_test_ratio = train_test_ratio
        self.random_seed = random_seed
        
        # Paths to important directories
        self.image_dir = os.path.join(root_dir, 'imagery', 'realsense_overhead')
        self.metadata_paths = [
            os.path.join(root_dir, 'metadata', 'dish_metadata_cafe1.csv'),
            os.path.join(root_dir, 'metadata', 'dish_metadata_cafe2.csv')
        ]
        
        # Load and process the data
        self._load_data()
        
    def _load_metadata(self, path):
        """Helper function to load metadata with variable columns"""
        # Read the CSV file line by line
        with open(path, 'r') as f:
            lines = f.readlines()
        
        # Process each line to extract the fixed columns we need
        metadata = []
        for line in lines[1:]:  # skip header
            parts = line.strip().split(',')
            dish_id = parts[0]
            
            # Extract the fixed nutritional values (columns 1-5 after dish_id)
            try:
                nutritional_values = list(map(float, parts[1:6]))
                metadata.append([dish_id] + nutritional_values)
            except (ValueError, IndexError):
                continue
                
        # Create DataFrame with just the columns we need
        columns = ['dish_id', 'total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein']
        return pd.DataFrame(metadata, columns=columns)
    
    def _load_data(self):
        """Load and process the dataset"""
        # Get all available dish IDs from the image directory
        all_dish_ids = [d for d in os.listdir(self.image_dir) if os.path.isdir(os.path.join(self.image_dir, d))]
        
        # Randomly sample n_samples dish IDs
        random.seed(self.random_seed)
        sampled_dish_ids = random.sample(all_dish_ids, min(self.n_samples, len(all_dish_ids)))
        
        # Load metadata from both cafes using our custom loader
        metadata_df = pd.concat([self._load_metadata(path) for path in self.metadata_paths])
        
        # Prepare image paths and labels
        self.samples = []
        for dish_id in sampled_dish_ids:
            # Get image paths
            dish_dir = os.path.join(self.image_dir, dish_id)
            rgb_path = os.path.join(dish_dir, 'rgb.png')
            depth_path = os.path.join(dish_dir, 'depth_color.png')
            
            # Check if both images exist
            if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
                continue
                
            # Get nutritional labels from metadata
            dish_metadata = metadata_df[metadata_df['dish_id'] == dish_id]
            if len(dish_metadata) == 0:
                continue
                
            # Extract the nutritional values
            nutritional_values = dish_metadata.iloc[0][[
                'total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein'
            ]].values.astype(float)
            
            self.samples.append({
                'dish_id': dish_id,
                'rgb_path': rgb_path,
                'depth_path': depth_path,
                'nutritional_values': nutritional_values
            })
        
        # Split into train and test sets
        train_samples, test_samples = train_test_split(
            self.samples,
            train_size=self.train_test_ratio,
            random_state=self.random_seed
        )
        
        self.samples = train_samples if self.split == 'train' else test_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load images
        rgb_image = Image.open(sample['rgb_path'])
        depth_image = Image.open(sample['depth_path'])
        
        # Apply transforms if specified
        if self.transform:
            rgb_image = self.transform(rgb_image)
            depth_image = self.transform(depth_image)
        
        # Get nutritional labels
        nutritional_values = torch.tensor(sample['nutritional_values'], dtype=torch.float32)
        
        return {
            'dish_id': sample['dish_id'],
            'rgb_image': rgb_image,
            'depth_image': depth_image,
            'nutritional_values': nutritional_values
        }

# Example usage remains the same
if __name__ == '__main__':
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    root_dir = '/Users/liwuchen/Documents/nutrition5k_dataset'
    train_dataset = Nutrition5KDataset(
        root_dir=root_dir,
        n_samples=1000,
        transform=transform,
        split='train',
        train_test_ratio=0.7
    )
    
    test_dataset = Nutrition5KDataset(
        root_dir=root_dir,
        n_samples=1000,
        transform=transform,
        split='test',
        train_test_ratio=0.7
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Example of iterating through the data
    for batch in train_loader:
        print(f"Batch RGB images shape: {batch['rgb_image'].shape}")
        print(f"Batch depth images shape: {batch['depth_image'].shape}")
        print(f"Batch nutritional values shape: {batch['nutritional_values'].shape}")
        break