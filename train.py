import torch
import torch.nn as nn
import torch.optim as optim
from cnn_model import NutritionCNN
from tqdm import tqdm
import logging

def train_model(train_loader, test_loader, num_epochs=20, learning_rate=0.001, device='cpu'):
    # Initialize model, loss function, and optimizer
    model = NutritionCNN().to(device)
    logging.info(f"Model loaded on {device}")
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop with tqdm for epochs
    for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
        model.train()
        running_loss = 0.0
        
        # Use tqdm for the training batches
        train_loop = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in train_loop:
            # Move data to device
            rgb_images = batch['rgb_image'].to(device)
            depth_images = batch['depth_image'].to(device)
            nutritional_values = batch['nutritional_values'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(rgb_images, depth_images)
            loss = criterion(outputs, nutritional_values)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Update progress bar with current loss
            train_loop.set_postfix(loss=loss.item())
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        # Use tqdm for the test batches
        test_loop = tqdm(test_loader, leave=False, desc="Testing")
        with torch.no_grad():
            for batch in test_loop:
                rgb_images = batch['rgb_image'].to(device)
                depth_images = batch['depth_image'].to(device)
                nutritional_values = batch['nutritional_values'].to(device)
                
                outputs = model(rgb_images, depth_images)
                loss = criterion(outputs, nutritional_values).item()
                test_loss += loss
                
                # Update progress bar with current loss
                test_loop.set_postfix(loss=loss)
        
        avg_test_loss = test_loss / len(test_loader)
        
        # Print epoch summary
        tqdm.write(f'\nEpoch [{epoch+1}/{num_epochs}], '
                   f'Train Loss: {avg_train_loss:.4f}, '
                   f'Test Loss: {avg_test_loss:.4f}\n')
    
    return model