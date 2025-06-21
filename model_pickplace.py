import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN

import torch
import torch.nn as nn
import torch.nn.functional as F

class InstancePointNet(nn.Module):
    def __init__(self, feat_dim=64, max_instances=20):
        super(InstancePointNet, self).__init__()
        self.feat_dim = feat_dim
        self.max_instances = max_instances  # Max number of instances to predict

        # Point-wise feature extraction (shared MLP)
        self.mlp1 = nn.Linear(3, 64)
        self.mlp2 = nn.Linear(64, 128)
        self.mlp3 = nn.Linear(128, feat_dim)

        # Instance Mask Head (predicts per-instance masks)
        self.mask_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feat_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)  # Mask logit per point
            ) for _ in range(max_instances)
        ])

        # Detection Heads
        self.centroid_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # centroid (x, y, z)
        )

        self.bbox_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 24),  # 8 corners × 3 coords
        )

    def forward(self, pc):  # pc: [P, 3]
        x = F.relu(self.mlp1(pc))
        x = F.relu(self.mlp2(x))
        features = self.mlp3(x)  # [P, feat_dim]

        # Global feature for detection
        global_feat = torch.max(features, dim=0, keepdim=True)[0]  # [1, feat_dim]

        instance_preds = []
        for i in range(self.max_instances):
            # Predict per-point logits for this instance
            mask_logits = self.mask_head[i](features).squeeze(-1)  # [P]

            # Optionally: mask global features using predicted logits
            masked_feat = torch.sum(
                features * mask_logits.unsqueeze(-1).sigmoid(), dim=0, keepdim=True
            ) / (mask_logits.sigmoid().sum() + 1e-6)

            centroid = self.centroid_head(masked_feat).squeeze(0)  # [3]
            bbox = self.bbox_head(masked_feat).view(8, 3)          # [8, 3]

            instance_preds.append({
                'mask_logit': mask_logits,
                'centroid': centroid,
                'bbox': bbox
            })

        return {
            'instance_preds': instance_preds  # List of dicts with mask/centroid/bbox
        }

if __name__ == '__main__':

    from dataset_pickplace import get_dataloaders


    import torch, random
    import numpy as np

    # Use MPS device if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Instantiate and move model to MPS
    model = InstancePointNet().to(device)
    model.eval()

    data_root = 'data/pick_place'
    train_loader, val_loader, test_loader = get_dataloaders(data_root)

    for batch in train_loader:
        pc = batch['point_cloud']   # [B, Points, 3]
        mask = batch['mask']        # [B, N, Points]
        bbox = batch['bbox3d']      # [B, N, 8, 3]
        centroid = batch['centroid']# [B, N, 3]

        # Use only the first sample in the batch
        pc_sample = pc[0]           # [Points, 3]
        mask_sample = mask[0]       # [N, Points]
        bbox_sample = bbox[0]       # [N, 8, 3]
        centroid_sample = centroid[0] # [N, 3]

        print("point_cloud shape:", np.array(pc_sample).shape)
        print("mask shape:", np.array(mask_sample).shape)
        print("bbox3d shape:", np.array(bbox_sample).shape)
        print("centroid shape:", np.array(centroid_sample).shape)

        # Optional: still visualize using NumPy
        #visualize_sample_plotly(pc_sample, mask_sample, bbox_sample)

        # ✅ Convert to Tensor and move to MPS device
        pc_tensor = torch.tensor(pc_sample, dtype=torch.float32).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(pc_tensor)

        print("Number of predicted instances:", len(outputs['instance_preds']))
        for i, inst in enumerate(outputs['instance_preds']):
            print(f"\nInstance {i}:")
            print("  mask_logit:", inst['mask_logit'].shape)   # [Points]
            print("  centroid:", inst['centroid'].shape)       # [3]
            print("  bbox:", inst['bbox'].shape)               # [8, 3]

        break

    

    import torch
    import torch.nn as nn
    import torch.optim as optim

    # Load model and move to device
    device = torch.device("mps" if torch.cuda.is_available() else "mps")
    model = InstancePointNet(max_instances=25).to(device)  # Set to your expected max N

    # Loss functions
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # === Get One Sample ===
    data_root = 'data/pick_place'
    train_loader, _, _ = get_dataloaders(data_root)

    for batch in train_loader:
        pc_sample = torch.from_numpy(batch['point_cloud'][0]).float().to(device)
        mask_sample = torch.from_numpy(batch['mask'][0]).float().to(device)
        bbox_sample = torch.from_numpy(batch['bbox3d'][0]).float().to(device)
        centroid_sample = torch.from_numpy(batch['centroid'][0]).float().to(device)
        break  # Only use the first sample

    # Move to CPU and package in a dictionary
    sample = {
        'point_cloud': pc_sample.cpu(),
        'mask': mask_sample.cpu(),
        'bbox3d': bbox_sample.cpu(),
        'centroid': centroid_sample.cpu()
    }

    # Save to a single .pt file
    torch.save(sample, 'first_sample.pt')

    num_instances = mask_sample.shape[0]  # N
    pc_sample = pc_sample.float()

    # === Training Loop ===
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()

        output = model(pc_sample)  # pc_sample: [Points, 3]
        preds = output['instance_preds']  # List of N_pred items

        total_loss = 0.0
        for i in range(num_instances):
            pred = preds[i]

            # Mask loss
            mask_logit = pred['mask_logit']         # [Points]
            gt_mask = mask_sample[i]                # [Points]
            loss_mask = bce_loss(mask_logit, gt_mask)

            # Centroid loss
            pred_centroid = pred['centroid']        # [3]
            gt_centroid = centroid_sample[i]        # [3]
            loss_centroid = mse_loss(pred_centroid, gt_centroid)

            # BBox loss
            pred_bbox = pred['bbox']                # [8, 3]
            gt_bbox = bbox_sample[i]                # [8, 3]
            loss_bbox = mse_loss(pred_bbox, gt_bbox)

            # Combine
            loss = loss_mask + loss_centroid + loss_bbox
            total_loss += loss

            print(f"Total Loss {total_loss}")

        total_loss.backward()
        optimizer.step()
        print(f"[Epoch {epoch}] Loss: {total_loss.item():.4f}")

        if epoch % 50 == 0:
            print(f"[Epoch {epoch}] Loss: {total_loss.item():.4f}")
            # Save model weights after training
            save_path = "instance_pointnet_overfit.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Model weights saved to: {save_path}")

    
    # Run inference
    with torch.no_grad():
        output = model(pc_sample)
        preds = output['instance_preds']

    torch.save(preds.cpu(), 'instance_preds.pt')
    num_instances = mask_sample.shape[0]

    for i in range(num_instances):
        pred = preds[i]

        # Predicted mask (apply sigmoid to logits)
        pred_mask = torch.sigmoid(pred['mask_logit']) > 0.5  # binary mask

        # Ground truth mask
        gt_mask = mask_sample[i] > 0.5

        # Predicted centroid and bbox
        pred_centroid = pred['centroid']
        pred_bbox = pred['bbox']

        # Ground truth centroid and bbox
        gt_centroid = centroid_sample[i]
        gt_bbox = bbox_sample[i]

        # Calculate Mask IoU
        intersection = (pred_mask & gt_mask).sum().item()
        union = (pred_mask | gt_mask).sum().item()
        iou = intersection / union if union > 0 else 0

        # Bbox difference (mean L2 distance)
        bbox_diff = torch.norm(pred_bbox - gt_bbox, dim=1).mean().item()

        print(f"\nInstance {i}:")
        print(f" - Mask IoU: {iou:.4f}")
        print(f" - Centroid Prediction: {pred_centroid.cpu().numpy()}")
        print(f" - Centroid Ground Truth: {gt_centroid.cpu().numpy()}")
        print(f" - BBox Mean L2 Diff: {bbox_diff:.4f}")

