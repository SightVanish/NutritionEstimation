import torch
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from model import RGBDFusionNetwork  # Replace with your model file
from loss import NutritionLoss  # Replace with your loss function file
from data_loader import get_nutrition5k_dataset, get_dataloader  # Replace with your dataset loading functions

def main():
    # Set the root directory and label file paths
    root_dir = 'data/nutrition5k_dataset/imagery'  # Replace with your actual path
    label_file = 'label_train.txt'

    # Set device to MPS if available, else CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS device for training.")
    else:
        device = torch.device('cpu')
        print("MPS device not found. Using CPU.")

    # Hyperparameters
    num_epochs = 100
    batch_size = 16
    learning_rate = 1e-3
    patience = 10  # Early stopping patience
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Get the dataset and dataloader
    dataset = get_nutrition5k_dataset(root_dir=root_dir, label_file=label_file)
    dataloader = get_dataloader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = RGBDFusionNetwork()
    model.to(device)

    loss_fn = NutritionLoss(num_tasks=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # TensorBoard setup
    writer = SummaryWriter(log_dir="logs")

    # Early stopping and best model tracking
    best_val_loss = float('inf')
    early_stop_counter = 0

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Training loop
        for batch_idx, (rgb_inputs, depth_inputs, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
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
        train_loss = running_loss / len(dataloader.dataset)
        writer.add_scalar('Loss/Train', train_loss, epoch + 1)

        # Validation (replace with your validation loader)
        val_loss = validate_model(model, dataloader, loss_fn, device)
        writer.add_scalar('Loss/Validation', val_loss, epoch + 1)

        # Save the best model weights
        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving best model...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # Save the last model weights after each epoch
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "last_model.pth"))

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}")

        # Early stopping
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    writer.close()
    print("Training completed successfully. Best model saved as 'best_model.pth' and last model as 'last_model.pth'.")

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