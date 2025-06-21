import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def visualize_sample_plotly(pc, mask, bbox3d, show_background=False):
    """
    Interactive Plotly visualization of 3D point cloud with instance masks, centroids, and 3D bounding boxes.

    Args:
        pc: [P, 3] point cloud
        mask: [N, P] binary masks
        bbox3d: [N, 8, 3] bounding boxes with 8 corner points
        show_background: bool, whether to display unlabeled points (background)
    """
    pc = np.asarray(pc)
    mask = np.asarray(mask)
    bbox3d = np.asarray(bbox3d)

    P = pc.shape[0]
    N = mask.shape[0]

    # Assign each point its instance ID
    instance_ids = np.full((P,), -1)
    for idx in range(N):
        instance_ids[mask[idx] > 0] = idx

    # Color map
    colorscale = px.colors.qualitative.Dark24  # 24 distinct colors
    num_colors = len(colorscale)

    fig = go.Figure()

    # Add point cloud per instance
    centroids = []
    for i in range(N):
        inds = np.where(instance_ids == i)[0]
        if inds.size == 0:
            centroids.append(np.array([np.nan, np.nan, np.nan]))
            continue
        pts = pc[inds]
        cent = pts.mean(axis=0)
        centroids.append(cent)

        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers',
            marker=dict(size=2, color=colorscale[i % num_colors]),
            showlegend=False  # Remove legend entries for points
        ))

        # Add centroid point
        fig.add_trace(go.Scatter3d(
            x=[cent[0]], y=[cent[1]], z=[cent[2]],
            mode='markers',
            marker=dict(size=7, color='black', symbol='x'),
            showlegend=False  # No legend for centroids
        ))

    centroids = np.vstack(centroids)  # Shape: [N, 3]

    # Add unlabeled/background points if enabled
    if show_background:
        bg_inds = np.where(instance_ids == -1)[0]
        if bg_inds.size > 0:
            pts = pc[bg_inds]
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers',
                marker=dict(size=1, color='gray', opacity=0.3),
                showlegend=False  # Remove legend for background points
            ))

    # Draw bounding boxes
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for i, box in enumerate(bbox3d):
        for s, e in edges:
            x, y, z = zip(box[s], box[e])
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False  # Remove legend for bbox lines
            ))

    fig.update_layout(
        title='3D Point Cloud with Instance Masks, Centroids, and BBoxes',
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data',
        ),
        showlegend=False  # Remove all legends
    )

    fig.show()

