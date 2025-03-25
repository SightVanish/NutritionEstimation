import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import Nutrition5KDataset
from train import train_model
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

if __name__ == '__main__':
    device = torch.accelerator.current_accelerator()
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    logging.info("Loading datasets...")
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
    logging.info("Done loading datasets")
    
    logging.info("Model training...")
    # Train the model
    model = train_model(train_loader, test_loader, num_epochs=20, device=device)
    logging.info("Model training complete")
    
    # Save the model
    logging.info("Saving model...")
    torch.save(model.state_dict(), 'nutrition_cnn.pth')
    logging.info("Model saved in 'nutrition_cnn.pth'")