import torch
import torch.nn as nn

class NutritionLoss(nn.Module):
    """Custom Loss Function for Food Nutrition Estimation."""
    def __init__(self, num_tasks=5, epsilon=1e-6):
        super(NutritionLoss, self).__init__()
        self.num_tasks = num_tasks
        self.epsilon = epsilon  # Small constant to avoid division by zero

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Tensor of shape (batch_size, num_tasks), model outputs.
            targets: Tensor of shape (batch_size, num_tasks), ground truth values.
        Returns:
            loss: Scalar, total normalized MAE loss.
        """
        # Calculate per-task normalized MAE
        task_losses = []
        for i in range(self.num_tasks):
            # Select predictions and targets for the current task
            pred_task = predictions[:, i]
            target_task = targets[:, i]
            
            # Mean value of the target to normalize MAE
            target_mean = torch.mean(target_task) + self.epsilon  # Avoid division by zero
            
            # Normalized MAE for the current task
            task_loss = torch.mean(torch.abs(pred_task - target_task) / target_mean)
            task_losses.append(task_loss)
        
        # Average loss across all tasks
        total_loss = torch.mean(torch.stack(task_losses))
        return total_loss

# Example Usage
if __name__ == "__main__":
    # Dummy Predictions and Targets (Batch size = 4, Number of Tasks = 5)
    predictions = torch.tensor([
        [100.0, 200.0, 10.0, 20.0, 15.0],
        [120.0, 210.0, 11.0, 18.0, 16.0],
        [130.0, 220.0, 12.0, 19.0, 14.0],
        [110.0, 205.0, 10.5, 21.0, 15.5]
    ], dtype=torch.float32)

    targets = torch.tensor([
        [110.0, 195.0, 12.0, 22.0, 16.0],
        [115.0, 200.0, 11.5, 20.0, 15.5],
        [125.0, 215.0, 13.0, 18.5, 14.5],
        [112.0, 198.0, 10.8, 20.5, 15.8]
    ], dtype=torch.float32)

    # Instantiate Loss Function
    loss_fn = NutritionLoss(num_tasks=5)
    loss = loss_fn(predictions, targets)
    print(f"Loss: {loss.item()}")