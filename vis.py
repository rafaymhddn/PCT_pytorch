import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def visualize_sample(pc, masks, bboxes):
    """
    Visualize a single sample with its instance masks and bounding boxes.
    
    Args:
        pc (np.ndarray): Point cloud array of shape (H, W, 3).
        masks (np.ndarray): Mask array of shape (N, H, W), where N is the number of instances.
        bboxes (np.ndarray): Bounding boxes array of shape (N, 8, 3).
    """
    
    # Prepare a color palette for the different instances.
    num_instances = masks.shape[0]
    # We use a colormap to get distinct colors for each instance.
    colors_palette = plt.cm.get_cmap('tab10', num_instances)
    
    scene_points_list = []
    scene_colors_list = []
    all_bbox_geometries = []
    
    # Iterate through each instance to color its points and create its bounding box.
    for i in range(num_instances):
        # Create a boolean mask for the current instance.
        instance_mask = masks[i] > 0  # Shape: (H, W)
        
        # Use the boolean mask to select points from the point cloud.
        instance_points = pc[instance_mask]  # Shape: (num_points_in_instance, 3)
        
        # Skip if the current mask has no corresponding points.
        if len(instance_points) == 0:
            continue
            
        # Assign a unique color to this instance's points.
        color = colors_palette(i)[:3]  # Get RGB values from the colormap.
        instance_colors = np.tile(color, (instance_points.shape[0], 1))
        
        scene_points_list.append(instance_points)
        scene_colors_list.append(instance_colors)
        
        # Define the edges for an Open3D bounding box.
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges connecting faces
        ]
        
        # Create an Open3D LineSet object for the bounding box.
        box_geometry = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(bboxes[i]),
            lines=o3d.utility.Vector2iVector(edges)
        )
        box_geometry.paint_uniform_color(color)
        all_bbox_geometries.append(box_geometry)
    
    # If no points were found in any instance, print a message and exit.
    if not scene_points_list:
        print("Warning: No points found for any instance masks. Nothing to visualize.")
        return
    
    # Combine the points and colors from all instances into single arrays.
    scene_points = np.concatenate(scene_points_list, axis=0)
    scene_colors = np.concatenate(scene_colors_list, axis=0)
    
    # Create the final Open3D PointCloud object.
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_points)
    scene_pcd.colors = o3d.utility.Vector3dVector(scene_colors)
    
    # Visualize the point cloud and all bounding boxes together.
    #print(f"Visualizing {len(all_bbox_geometries)} instances...")
    #print(f"Total points in scene: {len(scene_points)}")
    o3d.visualization.draw_geometries(
        [scene_pcd] + all_bbox_geometries,
        window_name="Segmented Point Cloud with Bounding Boxes"
    )

