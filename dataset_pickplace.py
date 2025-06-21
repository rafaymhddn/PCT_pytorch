import os
import glob
import numpy as np
from torch.utils.data import Dataset
from vis import *
from sklearn.model_selection import train_test_split


def custom_collate(batch):
    return {
        'point_cloud': [item['point_cloud'] for item in batch],
        'mask': [item['mask'] for item in batch],
        'bbox3d': [item['bbox3d'] for item in batch]
    }

class PickPlaceDataset(Dataset):
    def __init__(self, root_dir, sample_ids,  augment=False):
        self.root_dir = root_dir
        self.sample_ids = sample_ids
        self.augment = augment

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        sample_path = os.path.join(self.root_dir, sample_id)

        pc = np.load(os.path.join(sample_path, 'pc.npy'))        # [3, H, W]
        mask = np.load(os.path.join(sample_path, 'mask.npy'))    # [N, H, W]
        bbox = np.load(os.path.join(sample_path, 'bbox3d.npy'))  # [B, 7] or similar

        if pc.shape[0] == 3 and pc.shape[1] != 3:
            pc = pc.T # [3, W, H ]
            mask = mask.transpose(0, 2, 1)  # Now shape is [N, W, H]

        if self.augment:
            pc, bbox = self.apply_augmentations(pc, bbox)
        
        return {
            'point_cloud': pc.astype(np.float32),
            'mask': mask.astype(np.int64),
            'bbox3d': bbox.astype(np.float32)
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
        
        all_sample_ids = sorted(os.listdir(data_root))
        all_sample_ids = [s for s in all_sample_ids if os.path.isdir(os.path.join(data_root, s))]

        random.seed(seed)
        random.shuffle(all_sample_ids)

        train_ids, temp_ids = train_test_split(
        all_sample_ids, test_size=(1-train_size), random_state=seed
        )
    
        # test and val split 50:50
        val_ids, test_ids = train_test_split(
        temp_ids, test_size=(0.5), random_state=seed
        )

        train_set = PickPlaceDataset(data_root, train_ids,  augment=True)
        val_set = PickPlaceDataset(data_root, val_ids)
        test_set = PickPlaceDataset(data_root, test_ids)

        print(f"Dataset sizes - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

        return train_loader, val_loader, test_loader


if __name__ == '__main__':


    import os
    import random
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split

    
    
    # Testing
    
    data_root = 'data/pick_place'
    train_loader, val_loader, test_loader = get_dataloaders(data_root)

    for batch in train_loader:
        pc = batch['point_cloud']  # [H, W, 3]
        mask = batch['mask']       # [N, H, W]
        bbox = batch['bbox3d']     # [N, 8, 3] 

        print("point_cloud shape:", np.array(pc[0]).shape)
        print("mask shape:", np.array(mask[0]).shape)
        print("bbox3d shape:", np.array(bbox[0]).shape)

        visualize_sample_mat(pc[0], mask[0], bbox[0])
       
    
        break


