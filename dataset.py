import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from vis import *


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    ZIP_PATH = os.path.join(BASE_DIR, 'modelnet40_ply_hdf5_2048.zip')
    TARGET_DIR = os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(TARGET_DIR):
        if os.path.exists(ZIP_PATH):
            # Unzip directly into the data directory
            os.system(f'unzip -q {ZIP_PATH} -d {DATA_DIR}')
        else:
            raise FileNotFoundError(f"{ZIP_PATH} not found. Please place modelnet40_ply_hdf5_2048.zip in the project root.")


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')  # you can modify here to assign the path where dataset's root located at
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

# Data loader Test

import os
import random
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def custom_collate(batch):
    return {
        'point_cloud': [item['point_cloud'] for item in batch],
        'mask': [item['mask'] for item in batch],
        'bbox3d': [item['bbox3d'] for item in batch]
    }

class PickPlaceDataset(Dataset):
    def __init__(self, root_dir, sample_ids):
        self.root_dir = root_dir
        self.sample_ids = sample_ids

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        sample_path = os.path.join(self.root_dir, sample_id)

        pc = np.load(os.path.join(sample_path, 'pc.npy'))        # [N, 3]
        mask = np.load(os.path.join(sample_path, 'mask.npy'))    # [N]
        bbox = np.load(os.path.join(sample_path, 'bbox3d.npy'))  # [B, 7] or similar

        return {
            'point_cloud': pc.astype(np.float32),
            'mask': mask.astype(np.int64),
            'bbox3d': bbox.astype(np.float32)
        }

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

        train_set = PickPlaceDataset(data_root, train_ids)
        val_set = PickPlaceDataset(data_root, val_ids)
        test_set = PickPlaceDataset(data_root, test_ids)

        #print(f"Dataset sizes - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

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
        pc = batch['point_cloud']  # [3, H, W]
        mask = batch['mask']       # [N, H, W]
        bbox = batch['bbox3d']     # [N, 8, 3] 

        print("point_cloud shape:", np.array(pc[0]).shape)
        print("mask shape:", np.array(mask[0]).shape)
        print("bbox3d shape:", np.array(bbox[0]).shape)

        visualize_sample(pc[0], mask[0], bbox[0])
       
    
        break


