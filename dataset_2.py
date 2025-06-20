import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def visualize_sample(pc, masks, bboxes):
    """
    Visualize a single sample from the batch
    
    Args:
        pc: Point cloud array of shape (3, H, W)
        masks: Mask array of shape (N, H, W) 
        bboxes: Bounding boxes array of shape (N, 8, 3)
    """
    # Convert pc to shape (H, W, 3)
    #pc = np.transpose(pc, (1, 2, 0))  # (H, W, 3)
    #H, W, _ = pc.shape
    
    # Prepare colors for each instance
    colors_palette = plt.cm.get_cmap('tab10', masks.shape[0])
    scene_points = []
    scene_colors = []
    all_bboxes = []
    
    # Add all points and color by instance
    for i in range(masks.shape[0]):
        mask = masks[i]
        instance_points = pc[mask > 0]
        if len(instance_points) == 0:
            continue
            
        color = colors_palette(i)[:3]
        instance_colors = np.tile(color, (instance_points.shape[0], 1))
        
        scene_points.append(instance_points)
        scene_colors.append(instance_colors)
        
        # Create bounding box LineSet
        edges = [
            [0,1], [1,2], [2,3], [3,0],  # bottom face
            [4,5], [5,6], [6,7], [7,4],  # top face
            [0,4], [1,5], [2,6], [3,7]   # vertical edges
        ]
        box = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(bboxes[i]),
            lines=o3d.utility.Vector2iVector(edges)
        )
        box.paint_uniform_color(color)
        all_bboxes.append(box)
    
    # Check if we have any valid points
    if len(scene_points) == 0:
        print("No valid points found in any mask!")
        return
    
    # Combine all instance points
    scene_points = np.concatenate(scene_points, axis=0)
    scene_colors = np.concatenate(scene_colors, axis=0)
    
    # Create final point cloud
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_points)
    scene_pcd.colors = o3d.utility.Vector3dVector(scene_colors)
    
    # Visualize everything together
    print(f"Visualizing {masks.shape[0]} instances in a unified point cloud...")
    print(f"Total points: {len(scene_points)}")
    o3d.visualization.draw_geometries([scene_pcd] + all_bboxes, window_name="Scene with All Instances")

# Usage with your batch data:
# pc = batch['point_cloud']  # [B, 3, H, W] or [3, H, W]
# mask = batch['mask']       # [B, N, H, W] or [N, H, W]
# bbox = batch['bbox3d']     # [B, N, 8, 3] or [N, 8, 3]

# For a single sample from batch:
# visualize_sample(pc[0], mask[0], bbox[0])

# Or if your data is already single sample (no batch dimension):
# visualize_sample(pc, mask, bbox)

# NOTE: You might need a custom collate function if point clouds in a batch have different numbers of points.
# This is a common placeholder for such a function.
def custom_collate(batch):
    """
    Custom collate function to handle lists of tensors of varying sizes.
    This is a basic example; you might need to implement padding or other strategies.
    """
    keys = batch[0].keys()
    collated_batch = {key: [d[key] for d in batch] for key in keys}
    return collated_batch


class PickPlaceDataset(Dataset):
    """
    Dataset for loading pick and place data, with optional data augmentation.
    """
    def __init__(self, root_dir, sample_ids, augment=False):
        self.root_dir = root_dir
        self.sample_ids = sample_ids
        self.augment = augment

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        sample_path = os.path.join(self.root_dir, sample_id)

        pc = np.load(os.path.join(sample_path, 'pc.npy'))      # Can be [N, 3] or [3, N]
        mask = np.load(os.path.join(sample_path, 'mask.npy'))  # Shape: [N]
        bbox = np.load(os.path.join(sample_path, 'bbox3d.npy')) # Shape: [B, 8, 3]

        # --- FIX: Standardize point cloud to [N, 3] format ---
        # The error indicates the point cloud might be in [3, N] format.
        # We check for this and transpose it to the expected [N, 3] format.
        if pc.shape[0] == 3 and pc.shape[1] != 3:
            pc = pc.T
        
        # Now pc is guaranteed to be [N, 3], so augmentations will work correctly.
        if self.augment:
            pc, bbox = self.apply_augmentations(pc, bbox)

        return {
            'point_cloud': torch.from_numpy(pc.astype(np.float32)),
            'mask': torch.from_numpy(mask.astype(np.int64)),
            'bbox3d': torch.from_numpy(bbox.astype(np.float32))
        }

    def apply_augmentations(self, pc, bbox):
        """
        Applies a series of random augmentations to the point cloud and bounding box.
        Assumes pc is [N, 3] and bbox is [B, 8, 3].
        """
        # 1. Random Rotation around the Z-axis
        angle = random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1]
        ])
        pc = pc @ rotation_matrix.T
        bbox = bbox @ rotation_matrix.T

        # 2. Random Scaling
        scale = random.uniform(0.9, 1.1)
        pc *= scale
        bbox *= scale

        # 3. Random Jitter (translation)
        jitter = (np.random.rand(1, 3) - 0.5) * 0.02
        pc += jitter
        bbox += jitter
        
        return pc, bbox


def get_dataloaders(data_root, batch_size=16, seed=42, train_size=0.8):
    """
    Splits data into train, validation, and test sets and creates DataLoaders.
    Augmentation is applied only to the training set.
    """
    all_sample_ids = sorted(os.listdir(data_root))
    all_sample_ids = [s for s in all_sample_ids if os.path.isdir(os.path.join(data_root, s))]

    random.seed(seed)
    random.shuffle(all_sample_ids)

    train_ids, temp_ids = train_test_split(
        all_sample_ids, test_size=(1 - train_size), random_state=seed
    )
    
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=0.5, random_state=seed
    )

    train_set = PickPlaceDataset(data_root, train_ids, augment=True)
    val_set = PickPlaceDataset(data_root, val_ids, augment=False)
    test_set = PickPlaceDataset(data_root, test_ids, augment=False)

    print(f"Dataset sizes - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # --- Create a dummy dataset for demonstration ---
    print("Creating a dummy dataset for testing...")
    data_root = 'data/pick_place'
    if not os.path.exists(data_root):
        os.makedirs(data_root)
        print(f"Directory {data_root} created.")
    
    # Create 40 dummy samples, some [N, 3] and some [3, N] to test the fix
    if len(os.listdir(data_root)) < 40:
      print("Generating dummy files...")
      for i in range(40):
          sample_dir = os.path.join(data_root, f'sample_{i:04d}')
          os.makedirs(sample_dir, exist_ok=True)
          
          num_points = random.randint(500, 2000)
          
          # Alternate between [N, 3] and [3, N] formats
          if i % 2 == 0:
              pc = np.random.rand(num_points, 3) # [N, 3] format
          else:
              pc = np.random.rand(3, num_points) # [3, N] format
              
          mask = np.random.randint(0, 2, size=(num_points,))
          bbox_corners = np.random.rand(1, 8, 3)
          np.save(os.path.join(sample_dir, 'pc.npy'), pc)
          np.save(os.path.join(sample_dir, 'mask.npy'), mask)
          np.save(os.path.join(sample_dir, 'bbox3d.npy'), bbox_corners)
    print("Dummy dataset is ready.")
    # --- End of dummy dataset creation ---

    # Testing the DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(data_root, batch_size=4)
    
    print("\n--- Checking one batch from train_loader (with augmentation) ---")
    try:
        for batch in train_loader:
            pc_list = batch['point_cloud']
            mask_list = batch['mask']
            bbox3d_list = batch['bbox3d']
            #print("point_cloud shape (first item in batch):", pc_list[0].shape)
            print("point_cloud shape:", np.array(pc_list[0]).shape)
            print("mask shape:", np.array(mask_list[0]).shape)
            print("bbox3d shape:", np.array(bbox3d_list[0]).shape)
            print("Batch loaded successfully!")
            visualize_sample(pc_list[0], mask_list[0], bbox3d_list[0])
            break
    except Exception as e:
        print(f"An error occurred: {e}")

    print("\n--- Checking one batch from val_loader (without augmentation) ---")
    try:
        for batch in val_loader:
            pc_list = batch['point_cloud']
            mask_list = batch['mask']
            bbox3d_list = batch['bbox3d']
            #print("point_cloud shape (first item in batch):", pc_list[0].shape)
            print("point_cloud shape:", np.array(pc_list[0]).shape)
            print("mask shape:", np.array(mask_list[0]).shape)
            print("bbox3d shape:", np.array(bbox3d_list[0]).shape)
            print("Batch loaded successfully!")
            visualize_sample(pc_list[0], mask_list[0], bbox3d_list[0])
            break
    except Exception as e:
        print(f"An error occurred: {e}")