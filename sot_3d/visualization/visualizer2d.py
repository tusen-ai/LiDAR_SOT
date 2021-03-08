import matplotlib.pyplot as plt
import numpy as np
from sot_3d.data_protos import BBox


class Visualizer2D:
    def __init__(self, figsize):
        self.figure = plt.figure(figsize=figsize)
        plt.axis('equal')
        self.COLOR_MAP = {
            'pc': np.array([140, 140, 136]) / 256,
            'bbox0': np.array([4, 157, 217]) / 256,
            'bbox1': np.array([191, 4, 54]) / 256,    # red
            'bbox2': np.array([0, 0, 0]) / 256,       # black
            'bbox3': np.array([224, 133, 250]) / 256, 
            'bbox4': np.array([32, 64, 40]) / 256
        }
    
    def show(self):
        plt.show()
    
    def close(self):
        plt.close()
    
    def save(self, path):
        plt.savefig(path)
    
    def handler_pc(self, pc, color='pc'):
        vis_pc = np.asarray(pc)
        plt.scatter(vis_pc[:, 0], vis_pc[:, 1], marker='o', color=self.COLOR_MAP[color])
    
    def handler_box(self, box: BBox, message: str='', color='bbox1'):
        corners = np.array(BBox.box2corners2d(box))[:, :2]
        corners = np.concatenate([corners, corners[0:1, :2]])
        plt.plot(corners[:, 0], corners[:, 1], color=self.COLOR_MAP[color])
        plt.text(corners[0, 0] - 1, corners[0, 1] - 1, message, color=self.COLOR_MAP['bbox2'])