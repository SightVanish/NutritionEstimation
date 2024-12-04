import torch
from model import RGBDFusionNetwork  # Replace with your model file
from data_loader import get_nutrition5k_datasets, get_dataloader  # Replace with your dataset loading functions
from utils import get_device
import numpy as np

def main():
    # Set paths
    root_dir = 'data/nutrition5k_dataset'
    label_file = 'imagery/label.txt'
    checkpoint_path = "checkpoints/best_model.pth"

    # Task labels
    task_labels = ["Calories (kcal)", "Fat (g)", "Protein (g)", "Carbohydrates (g)"]

    # Set device
    device = get_device()

    # Load test dataset
    _, _, test_dataset = get_nutrition5k_datasets(root_dir, label_file, 0.0, 0.0)
    test_loader = get_dataloader(test_dataset, batch_size=16, shuffle=False)

    # Load model
    model = RGBDFusionNetwork()
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.to(device)
    model.eval()

    # Initialize accumulators for errors
    num_tasks = len(task_labels)  # Explicit number of tasks from the labels
    task_errors = [[] for _ in range(num_tasks)]  # Per-task absolute errors
    task_percentage_errors = [[] for _ in range(num_tasks)]  # Per-task percentage errors

    # Testing loop
    with torch.no_grad():
        for rgb_inputs, depth_inputs, labels in test_loader:
            rgb_inputs = rgb_inputs.to(device, dtype=torch.float32)
            depth_inputs = depth_inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            # Get model predictions
            outputs = model(rgb_inputs, depth_inputs)

            # Per-sample, per-task errors
            for i in range(labels.shape[1]):  # Iterate through each task
                abs_errors = torch.abs(outputs[:, i] - labels[:, i])
                percentage_errors = abs_errors / (labels[:, i] + 1e-6) * 100  # Avoid division by zero
                task_errors[i].extend(abs_errors.cpu().numpy())
                task_percentage_errors[i].extend(percentage_errors.cpu().numpy())

    # Calculate average metrics per task
    mean_absolute_errors = [np.mean(errors) for errors in task_errors]
    mean_percentage_errors = [np.mean(errors) for errors in task_percentage_errors]

    # Report results
    print("\nTesting Results:")
    for i, (mae, mpe) in enumerate(zip(mean_absolute_errors, mean_percentage_errors)):
        print(f"Task: {task_labels[i]}")
        print(f"  Mean Absolute Error (MAE): {mae:.4f}")
        print(f"  Mean Percentage Error (MPE): {mpe:.2f}%")

    # Optional: Save results for later analysis
    results = {
        'task_labels': task_labels,
        'mean_absolute_errors': mean_absolute_errors,
        'mean_percentage_errors': mean_percentage_errors
    }
    torch.save(results, "test_results.pth")


if __name__ == '__main__':
    main()