import torch
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from model import RGBDFusionNetwork, BaseResNetModel  # Replace with your model file
from loss import NutritionLoss  # Replace with your loss function file
from data_loader import get_train_val_loaders  # Use the updated function from data_loader
from utils import get_device


def main():
    # Set paths
    root_dir = 'data/nutrition5k_dataset/imagery/train'
    label_file = 'train.txt'
    checkpoint_dir = "checkpoints"
    log_dir = "logs"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Set device
    device = get_device()

    # Hyperparameters
    num_epochs = 100
    batch_size = 16
    learning_rate = 1e-3
    val_ratio = 0.2  # Validation split ratio

    # Load datasets and DataLoaders
    train_loader, val_loader = get_train_val_loaders(root_dir, label_file, batch_size, val_ratio)

    # Initialize model, loss function, and optimizer
    # model = RGBDFusionNetwork()
    model = BaseResNetModel()
    model.to(device)

    loss_fn = NutritionLoss(num_tasks=4)  # Adjust num_tasks if needed
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # TensorBoard setup
    writer = SummaryWriter(log_dir=log_dir)

    # Best model tracking
    best_val_loss = float('inf')

    print("Starting training...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for batch_idx, (rgb_inputs, depth_inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            rgb_inputs = rgb_inputs.to(device, dtype=torch.float32)
            depth_inputs = depth_inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(rgb_inputs, depth_inputs)

            # Compute loss
            loss = loss_fn(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * rgb_inputs.size(0)

        # Calculate average training loss
        train_loss = running_loss / len(train_loader.dataset)
        writer.add_scalar('Loss/Train', train_loss, epoch + 1)

        # Evaluation phase
        val_loss = validate_model(model, val_loader, loss_fn, device)
        writer.add_scalar('Loss/Validation', val_loss, epoch + 1)

        # Save the best model weights
        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving best model...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))

        # Save the last model weights after each epoch
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "last_model.pth"))

        print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}")

    writer.close()
    print("Training completed. Best model saved as 'best_model.pth' and last model as 'last_model.pth'.")


def validate_model(model, val_loader, loss_fn, device):
    """Validation loop."""
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for rgb_inputs, depth_inputs, labels in val_loader:
            rgb_inputs = rgb_inputs.to(device, dtype=torch.float32)
            depth_inputs = depth_inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            outputs = model(rgb_inputs, depth_inputs)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item() * rgb_inputs.size(0)

    val_loss /= len(val_loader.dataset)
    return val_loss


if __name__ == '__main__':
    main()