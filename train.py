import torch
import torch.nn as nn
from model import RGBDResNet
from data_loader import get_nutrition5k_dataset, get_dataloader  # Replace with your actual DataLoader script

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
    num_epochs = 10
    batch_size = 1
    learning_rate = 1e-4

    # Get the dataset and dataloader
    # Assuming your DataLoader returns (rgb_image, depth_image, labels)
    dataset = get_nutrition5k_dataset(root_dir=root_dir, label_file=label_file)
    dataloader = get_dataloader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = RGBDResNet(num_outputs=5)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (rgb_inputs, depth_inputs, labels) in enumerate(dataloader):
            rgb_inputs = rgb_inputs.to(device, dtype=torch.float32)
            depth_inputs = depth_inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(rgb_inputs, depth_inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * rgb_inputs.size(0)

            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}')

    print('Training completed successfully.')

if __name__ == '__main__':
    main()