# Toolkits

In the project, we have implemented several useful toolkits. Remember to install `sot_3d` if you want to use these toolkits elsewhere in other of your projects.

## 1. 3D Bounding Boxes

We have provided a data prototype for manipulating the 3D BBoxes: `sot_3d.data_protos.BBox`. To convert general data format to `BBox`, we provides the following interfaces:

```python
bbox = BBox.array2bbox([center_x, center_y, center_z, heading, length, width, height, score])
bbox = BBox.dict2bbox({
    'center_x': center_x, 
    'center_y': center_y, 
    'center_z': center_z, 
    'heading': heading, 
    'length': length, 
    'width': width, 
    'height': height, 
    'score': score
})
```

On the basis of `BBox`, we also provide a large range of methods for 3D BBox computation. Please look at [BBox File](../sot_3d/data_protos/bbox.py)

## 2. 2D BEV Visualization

We have provided a visualizer of tracking results in BEV, especially the point clouds and bounding boxes. The example usages are as follows:

```python
from sot_3d.data_protos import BBox
from sot_3d.visualization import Visualizer2D
import numpy as np

pc = ...                                     # initialize the point cloud as a N * 3 numpy array
bbox = ...                                   # initialize a bbox as a BBox instance

visualizer = Visualizer2D(figsize=(8, 8))    # initialize a visualizer
visualizer.handler_pc(pc, color='gray')      # input the pc
visualizer.handler_box(bbox, color='red')    # draw the bbox

visualizer.show()                            # show the image
visualizer.save(YOUR_PATH)                   # save the 2D image
visualizer.close()
```

The color mapping are stored in `../sot_3d/visualization/visualizer2d.py`

```python
self.COLOR_MAP = {
    'gray': np.array([140, 140, 136]) / 256,
    'light_blue': np.array([4, 157, 217]) / 256,
    'red': np.array([191, 4, 54]) / 256,
    'black': np.array([0, 0, 0]) / 256,
    'purple': np.array([224, 133, 250]) / 256, 
    'dark_green': np.array([32, 64, 40]) / 256
}
```

The requirements of this visualizer are

```
numpy matplotlib
```

