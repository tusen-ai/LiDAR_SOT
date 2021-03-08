# Toolkits

In the project, we have implemented several useful toolkits.

## 1 3D Bounding Boxes

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

## 2 2D Visualization

We have provided a visualizer of tracking results in BEV.