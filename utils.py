import os
import shutil
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch

class Nutrition_RGBD(Dataset):
    def __init__(self, image_path, rgb_txt_dir, rgbd_txt_dir, transform=None):
        with open(rgb_txt_dir, 'r') as file_rgb:
            lines_rgb = file_rgb.readlines()
        with open(rgbd_txt_dir, 'r') as file_rgbd:
            lines_rgbd = file_rgbd.readlines()

        self.images = []
        self.images_rgbd = []
        self.labels = []
        self.total_calories = []
        self.total_mass = []
        self.total_fat = []
        self.total_carb = []
        self.total_protein = []

        for line in lines_rgb:
            parts = line.strip().split()
            self.images.append(os.path.join(image_path, parts[0]))
            self.labels.append(str(parts[1]))
            self.total_calories.append(float(parts[2]))
            self.total_mass.append(float(parts[3]))
            self.total_fat.append(float(parts[4]))
            self.total_carb.append(float(parts[5]))
            self.total_protein.append(float(parts[6]))

        for line in lines_rgbd:
            self.images_rgbd.append(os.path.join(image_path, line.split()[0]))

        self.transform = transform

    def __getitem__(self, index):
        img_rgb = Image.open(self.images[index]).convert('RGB')
        img_rgbd = Image.open(self.images_rgbd[index]).convert('RGB')
        if self.transform is not None:
            img_rgb = self.transform(img_rgb)
            img_rgbd = self.transform(img_rgbd)

        return (
            img_rgb, self.labels[index], self.total_calories[index], 
            self.total_mass[index], self.total_fat[index], 
            self.total_carb[index], self.total_protein[index], img_rgbd
        )

    def __len__(self):
        return len(self.images)

def get_transforms(dataset, model=None):
    """Define data transformations with augmentations for the nutrition_rgb and nutrition_rgbd datasets."""
    
    normalization = [0.485, 0.456, 0.406] if model and 'vit' in model else [0.5, 0.5, 0.5]
    common_transform = [
        transforms.ToTensor(),
        transforms.Normalize(normalization, normalization)
    ]
    
    if dataset == 'nutrition_rgb':
        resize_size = (270, 480)
        crop_size = (224, 224) if model and 'vit' in model else (256, 256)
        
        train_transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(0, 15)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            *common_transform
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            *common_transform
        ])

    elif dataset == 'nutrition_rgbd':
        train_transform = transforms.Compose([
            transforms.Resize((320, 448)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(0, 15)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            *common_transform
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((320, 448)),
            *common_transform
        ])
    
    else:
        raise ValueError("Unsupported dataset type")
    
    return train_transform, test_transform

def get_dataset(args, train_transform, test_transform):
    """Load the appropriate dataset (nutrition_rgb or nutrition_rgbd) based on args."""
    if args.dataset == 'nutrition_rgb':
        image_path = os.path.join(args.data_root, 'imagery')
        train_txt = os.path.join(args.data_root, 'imagery', 'rgb_train_processed.txt')
        test_txt = os.path.join(args.data_root, 'imagery', 'rgb_test_processed.txt')

        trainset = Nutrition_RGBD(image_path=image_path, txt_dir=train_txt, transform=train_transform)
        testset = Nutrition_RGBD(image_path=image_path, txt_dir=test_txt, transform=test_transform)

    elif args.dataset == 'nutrition_rgbd':
        image_path = os.path.join(args.data_root, 'imagery')
        train_rgbd_txt = os.path.join(args.data_root, 'imagery', 'rgb_in_overhead_train_processed.txt')
        test_rgbd_txt = os.path.join(args.data_root, 'imagery', 'rgb_in_overhead_test_processed.txt')
        train_txt = os.path.join(args.data_root, 'imagery', 'rgbd_train_processed.txt')
        test_txt = os.path.join(args.data_root, 'imagery', 'rgbd_test_processed.txt')
        
        trainset = Nutrition_RGBD(image_path, train_rgbd_txt, train_txt, transform=train_transform)
        testset = Nutrition_RGBD(image_path, test_rgbd_txt, test_txt, transform=test_transform)
    else:
        raise ValueError("Unsupported dataset type")
    return trainset, testset

def get_device():
    """Get the device (GPU or CPU) based on availability."""
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS on MacOS.")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA on Nvidia.")
    else: 
        device = torch.device('cpu')
        print("CUDA and MPS devices not found. Using CPU.")
    return device

def split_dataset():
    # Define paths
    data_dir = "data/nutrition5k_dataset/imagery/realsense_overhead"
    label_file = "data/nutrition5k_dataset/imagery/label.txt"
    output_dir = "data/nutrition5k_dataset/imagery"
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    train_labels_file = os.path.join(output_dir, "train.txt")
    test_labels_file = os.path.join(output_dir, "test.txt")

    # Create directories for split datasets
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get all dish directories
    dish_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    # Shuffle the dish directories
    random.shuffle(dish_dirs)

    # Split the data (40% for testing)
    test_size = int(0.4 * len(dish_dirs))
    test_dishes = dish_dirs[:test_size]
    train_dishes = dish_dirs[test_size:]

    # Read the labels from the label file
    with open(label_file, "r") as f:
        labels = f.readlines()

    # Create dictionaries for labels
    label_dict = {}
    for line in labels:
        parts = line.strip().split(",")
        dish_id = parts[0]
        label_dict[dish_id] = line.strip()

    # Function to copy directories and create label files
    def copy_dishes_and_write_labels(dishes, dest_dir, label_file):
        with open(label_file, "w") as f:
            for dish in dishes:
                src_path = os.path.join(data_dir, dish)
                dest_path = os.path.join(dest_dir, dish)
                shutil.copytree(src_path, dest_path)
                # Write the corresponding label
                if dish in label_dict:
                    f.write(label_dict[dish] + "\n")

    # Copy training and testing data and write labels
    copy_dishes_and_write_labels(train_dishes, train_dir, train_labels_file)
    copy_dishes_and_write_labels(test_dishes, test_dir, test_labels_file)

    print("Dataset split completed.")
    print(f"Training samples: {len(train_dishes)}, Testing samples: {len(test_dishes)}")
    print(f"Labels written to {train_labels_file} and {test_labels_file}.")

def check_data_integrity():
    # Define file paths
    label_file = "data/nutrition5k_dataset/imagery/label.txt"
    data_directory = "data/nutrition5k_dataset/imagery/realsense_overhead"
    output_file = "data/nutrition5k_dataset/imagery/filtered_label.txt"

    # Open the label file and create a filtered output
    with open(label_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Split the line by commas
            parts = line.strip().split(',')
            
            # Extract the dish ID (assuming it's the first part of each line)
            dish_id = parts[0]
            
            # Check if a directory with the same name as the dish ID exists in the data directory
            if os.path.isdir(os.path.join(data_directory, dish_id)):
                # Write the valid line to the output file
                outfile.write(line)

    print(f"Filtered data has been written to {output_file}")

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
    
# def get_DataLoader(args):
#     """Main function to get data loaders for training and testing."""
#     train_transform, test_transform = get_transforms(args.dataset, model=args.model)
#     trainset, testset = get_dataset(args, train_transform, test_transform)

#     train_loader = DataLoader(trainset, batch_size=args.b, shuffle=True, num_workers=4, pin_memory=True)
#     test_loader = DataLoader(testset, batch_size=args.b, shuffle=False, num_workers=4, pin_memory=True)

#     return train_loader, test_loader
if __name__ == "__main__":
    # split_dataset()
    check_data_integrity()