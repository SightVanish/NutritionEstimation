import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

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



# def get_DataLoader(args):
#     """Main function to get data loaders for training and testing."""
#     train_transform, test_transform = get_transforms(args.dataset, model=args.model)
#     trainset, testset = get_dataset(args, train_transform, test_transform)

#     train_loader = DataLoader(trainset, batch_size=args.b, shuffle=True, num_workers=4, pin_memory=True)
#     test_loader = DataLoader(testset, batch_size=args.b, shuffle=False, num_workers=4, pin_memory=True)

#     return train_loader, test_loader
