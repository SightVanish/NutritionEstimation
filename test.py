import torch
from model import RGBDFusionNetwork  # Replace with your model file
from data_loader import get_nutrition5k_datasets, get_dataloader  # Replace with your dataset loading functions
from utils import get_device
import numpy as np
import shap  # Import SHAP for model explanation
import matplotlib.pyplot as plt

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

    # SHAP explainer requires a forward pass
    def model_predict(inputs):
        rgb_inputs, depth_inputs = inputs
        rgb_inputs = torch.tensor(rgb_inputs, dtype=torch.float32, device=device)
        depth_inputs = torch.tensor(depth_inputs, dtype=torch.float32, device=device)
        with torch.no_grad():
            predictions = model(rgb_inputs, depth_inputs).cpu().numpy()
        return predictions

    # Use a subset of the test set for SHAP explanations
    data_iter = iter(test_loader)
    rgb_inputs, depth_inputs, labels = next(data_iter)  # Get a single batch
    rgb_inputs = rgb_inputs.numpy()  # Convert to NumPy
    depth_inputs = depth_inputs.numpy()  # Convert to NumPy

    # Combine RGB and depth inputs for SHAP
    combined_inputs = (rgb_inputs, depth_inputs)

    # Create a SHAP explainer
    explainer = shap.Explainer(model_predict, combined_inputs)
    shap_values = explainer(combined_inputs)

    # Visualize explanations for the first prediction
    for task_idx, task_label in enumerate(task_labels):
        shap.waterfall_plot(shap.Explanation(values=shap_values[:, task_idx], 
                                             base_values=explainer.expected_value[task_idx],
                                             feature_names=["RGB", "Depth"]),
                            show=True)

    # Testing loop
    task_errors = [[] for _ in range(len(task_labels))]
    task_percentage_errors = [[] for _ in range(len(task_labels))]

    with torch.no_grad():
        for rgb_inputs, depth_inputs, labels in test_loader:
            rgb_inputs = rgb_inputs.to(device, dtype=torch.float32)
            depth_inputs = depth_inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            outputs = model(rgb_inputs, depth_inputs)

            for i in range(labels.shape[1]):
                abs_errors = torch.abs(outputs[:, i] - labels[:, i])
                percentage_errors = abs_errors / (labels[:, i] + 1e-6) * 100
                task_errors[i].extend(abs_errors.cpu().numpy())
                task_percentage_errors[i].extend(percentage_errors.cpu().numpy())

    mean_absolute_errors = [np.mean(errors) for errors in task_errors]
    mean_percentage_errors = [np.mean(errors) for errors in task_percentage_errors]

    print("\nTesting Results:")
    for i, (mae, mpe) in enumerate(zip(mean_absolute_errors, mean_percentage_errors)):
        print(f"Task: {task_labels[i]}")
        print(f"  Mean Absolute Error (MAE): {mae:.4f}")
        print(f"  Mean Percentage Error (MPE): {mpe:.2f}%")

    results = {
        'task_labels': task_labels,
        'mean_absolute_errors': mean_absolute_errors,
        'mean_percentage_errors': mean_percentage_errors
    }
    torch.save(results, "test_results.pth")

if __name__ == '__main__':
    main()