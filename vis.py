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