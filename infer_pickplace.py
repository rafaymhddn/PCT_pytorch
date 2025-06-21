import torch
import numpy as np
from model_pickplace import InstancePointNet
from dataset_pickplace import get_dataloaders
from vis import *

# from your_model_file import InstancePointNet  
device = torch.device("mps" if torch.cuda.is_available() else "mps")

# Initialize model and load weights
model = InstancePointNet(max_instances=25)
model.load_state_dict(torch.load("instance_pointnet_overfit.pth", map_location=device))
model.to(device)
model.eval()

# Load one sample from your dataloader
data_root = 'data/pick_place'
train_loader, _, _ = get_dataloaders(data_root)

for batch in train_loader:
    pc_sample = batch['point_cloud'][0]     # numpy array [Points, 3]
    mask_sample = batch['mask'][0]          # numpy array [N, Points]
    bbox_sample = batch['bbox3d'][0]        # numpy array [N, 8, 3]
    centroid_sample = batch['centroid'][0]  # numpy array [N, 3]
    break  # Use only the first sample

    visualize_sample_plotly(pc_sample[0], mask_sample[0], bbox_sample[0])

# Convert to torch tensors and send to device
pc_sample = torch.from_numpy(pc_sample).float().to(device)
mask_sample = torch.from_numpy(mask_sample).float().to(device)
bbox_sample = torch.from_numpy(bbox_sample).float().to(device)
centroid_sample = torch.from_numpy(centroid_sample).float().to(device)

# Run inference
with torch.no_grad():
    output = model(pc_sample)
    preds = output['instance_preds']

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

    # Visualize ground truth
    print("Visualizing Ground Truth:")
    visualize_sample_plotly(
        pc_sample.cpu().numpy(),
        gt_mask_np,
        gt_bbox_np
    )

    # Visualize prediction
    print("Visualizing Prediction:")
    visualize_sample_plotly(
        pc_sample.cpu().numpy(),
        pred_mask_np,
        pred_bbox_np
    )
        