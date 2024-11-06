from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from data_loader import Nutrition5kDataset

# Assume the Nutrition5kDataset class has been defined as provided earlier

# Set up the paths (replace with your actual paths)
root_dir = 'data/nutrition5k_dataset/imagery'  # Replace with your actual path
label_file = 'label_train.txt'

# Instantiate the dataset
dataset = Nutrition5kDataset(root_dir=root_dir, label_file=label_file, transform=True)

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

# Fetch a single batch
inputs, labels = next(iter(dataloader))

# Print the shapes of inputs and labels
print('Input tensor shape:', inputs.shape)  # Expected: [batch_size, 6, 224, 224]
print('Labels shape:', labels.shape)        # Expected: [batch_size, 5]

# Print the data types
print('Input tensor type:', inputs.dtype)   # Expected: torch.float32
print('Labels type:', labels.dtype)         # Expected: torch.float32

# Optionally, visualize the images
def imshow(input_tensor):
    # Unnormalize the image
    input_tensor = input_tensor.numpy()
    mean = np.array([0.485, 0.456, 0.406]*2).reshape(6, 1, 1)
    std = np.array([0.229, 0.224, 0.225]*2).reshape(6, 1, 1)
    input_tensor = std * input_tensor + mean  # Unnormalize
    input_tensor = np.clip(input_tensor, 0, 1)
    
    # Split the tensor into two images
    depth_color_image = input_tensor[:3, :, :]
    depth_raw_image = input_tensor[3:, :, :]
    
    # Transpose the images for matplotlib (C x H x W) -> (H x W x C)
    depth_color_image = np.transpose(depth_color_image, (1, 2, 0))
    depth_raw_image = np.transpose(depth_raw_image, (1, 2, 0))
    
    # Plot both images side by side
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(depth_color_image)
    axs[0].set_title('Depth Color Image')
    axs[0].axis('off')
    axs[1].imshow(depth_raw_image)
    axs[1].set_title('Depth Raw Image')
    axs[1].axis('off')
    plt.show()

# Visualize the first sample in the batch
imshow(inputs[0])