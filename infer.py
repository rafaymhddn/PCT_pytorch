import h5py

def load_one_example_from_h5(h5_filename, example_index=0):
    """
    Loads a single point cloud and its label from a ModelNet40 HDF5 file.

    Args:
        h5_filename (str): Path to the .h5 file.
        example_index (int): The index of the example to load from the file.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The point cloud data (shape: [num_points, 3]).
            - int: The label index.
    """
    with h5py.File(h5_filename, 'r') as f:
        point_cloud = f['data'][example_index]
        label = f['label'][example_index][0] # Labels are often stored as arrays, e.g., [25]
    return point_cloud, label

import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from model import NaivePCTCls, SPCTCls, PCTCls
from collections import OrderedDict

# --- Helper Dictionaries and Functions ---

models = {'navie_pct': NaivePCTCls,
          'spct': SPCTCls,
          'pct': PCTCls}

MODELNET40_CLASSES = {
    0: 'airplane', 1: 'bathtub', 2: 'bed', 3: 'bench', 4: 'bookshelf',
    5: 'bottle', 6: 'bowl', 7: 'car', 8: 'chair', 9: 'cone',
    10: 'cup', 11: 'curtain', 12: 'desk', 13: 'door', 14: 'dresser',
    15: 'flower_pot', 16: 'glass_box', 17: 'guitar', 18: 'keyboard',
    19: 'lamp', 20: 'laptop', 21: 'mantel', 22: 'monitor', 23: 'night_stand',
    24: 'person', 25: 'piano', 26: 'plant', 27: 'radio', 28: 'range_hood',
    29: 'sink', 30: 'sofa', 31: 'stairs', 32: 'stool', 33: 'table',
    34: 'tent', 35: 'toilet', 36: 'tv_stand', 37: 'vase', 38: 'wardrobe',
    39: 'xbox'
}

def load_one_example_from_h5(h5_filename, example_index=0):
    """Loads a single point cloud and its label from a ModelNet40 HDF5 file."""
    with h5py.File(h5_filename, 'r') as f:
        point_cloud = f['data'][example_index]
        label = f['label'][example_index][0]
    return point_cloud, label

# --- Main Inference and Visualization Function ---

def inference_and_visualize(point_cloud, model_path, model_type='pct', true_label_name="N/A"):
    """
    Perform inference on a single point cloud and visualize it.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    model = models[model_type]().to(device)
    try:
        # Try loading directly, works if saved/loaded on similar devices
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError:
        # Handles cases where the model was saved with nn.DataParallel
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    model.eval()

    # 2. Prepare Input
    point_cloud_tensor = torch.from_numpy(point_cloud).float().unsqueeze(0).permute(0, 2, 1).to(device)

    # 3. Perform Inference
    with torch.no_grad():
        logits = model(point_cloud_tensor)
        preds = logits.max(dim=1)[1]
        predicted_class_idx = preds.item()

    predicted_class_name = MODELNET40_CLASSES.get(predicted_class_idx, "Unknown")
    print(f"Predicted Class: '{predicted_class_name}' (Index: {predicted_class_idx})")

    # 4. Visualize the Point Cloud
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=15, c='dodgerblue')
    ax.set_title(f"True Label: '{true_label_name}'\nPredicted Label: '{predicted_class_name}'", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True)
    plt.show()


# --- Main execution block ---
if __name__ == "__main__":
    # --- Configuration ---
    # 1. SET THE PATH to your dataset files
    # Based on your image, this is the correct path structure.
    DATA_PATH = 'data/modelnet40_ply_hdf5_2048/ply_data_test0.h5'
    
    # 2. SET THE PATH to your trained model file (.t7)
    # This comes from your 'checkpoints' directory.
    MODEL_PATH = 'checkpoints/pct_cls/models/model.t7'
    MODEL_TYPE = 'spct'  # Or 'spct', 'navie_pct'

    # 3. CHOOSE WHICH EXAMPLE to load from the file
    EXAMPLE_INDEX = 71 # Let's pick the 51st object in the file

    # --- Execution ---
    try:
        # Load one point cloud and its true label
        sample_point_cloud, true_label_idx = load_one_example_from_h5(DATA_PATH, EXAMPLE_INDEX)
        true_label_name = MODELNET40_CLASSES.get(true_label_idx, "Unknown")

        print(f"Loading example {EXAMPLE_INDEX} from {DATA_PATH}")
        print(f"True Label: '{true_label_name}' (Index: {true_label_idx})")

        # Perform inference and visualize the result
        inference_and_visualize(sample_point_cloud, MODEL_PATH, MODEL_TYPE, true_label_name)

    except FileNotFoundError:
        print("-" * 50)
        print(f"ERROR: Could not find the data or model file.")
        print(f"Please check that the following paths are correct:")
        print(f"Data file: {DATA_PATH}")
        print(f"Model file: {MODEL_PATH}")
        print("-" * 50)
    except Exception as e:
        print(f"An error occurred: {e}")