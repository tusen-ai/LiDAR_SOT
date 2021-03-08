import open3d as o3d, numpy as np


__all__ = [
    'o3d_pc_visualization'
]


def o3d_pc_visualization(pc):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pcd])
    return