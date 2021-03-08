# Description of `configs.yaml`

The fields and the description of `configs.yaml` are as follows. Please look at the corresponding components to see how we may initialize the whole tracker using the configurations.

```yaml
running:                     # running configurations
  window_size: 2             # how many frames to consider 
    
data_loader:                 # data loader configurations
  terrain: false             # whether to load the terrain map
    
finder:                      # configurations for finding the motion between frame 0-1
  optim:                     # optimizer configurations
    method: BFGS             # optimizer type
    ITER: 50                 # iterations for iterations out-of-optimizer
    options:                 # optimizer related options
      maxiter: 50            # maximum iterations for optimizer
      disp: false    
  neighbor_num: 1            # nearest neighbor number for icp loss computation
  box_scaling_prev: 1.0      # when segmenting the related point cloud, the scale in the previous box    
  box_scaling_next: 3.0      # when segmenting the related point cloud, the scale in the next box    
  least_pc: 10               # least LiDAR point number to consider   
  pc_limit: 10000            # maximum LiDAR point number, downsample the number of surpass the limit    
  loss_type: L2              # loss type, such as L2, Huber
  ransac:                    # ransac options
    switch: false            # if use ransac
    
optim:                       # configurations for the optimizer in tracking
  method: BFGS               # optimizer type
  iter: 20                   # iteration out-og-optimizer
  options:                   # optimizer options
    maxiter: 50              # maximum optimizer iterations
    disp: false    
    
motion_model:                # motion model related configurations
  moving_avg_weight: 0.5     # moving average weight
    
shape_map:                   # configurations for maintaining a shape during tracking
  update_freq: 1000          # how many frames for an update
  subshape_len: 2            # subshape length
  box_scaling_pc_bank: 1.0   # the scale for segmenting the related LiDAR points
  downsample: true           # if downsample the shape
  resolution: 0.05           # the voxel size for downsampling

factors:                     # loss factors
  switch:                    # the switch for some factors, might save time
    latitude: false          # whether to use the latitude factor
  names:                     # factor names
    - latitude
    - icp_loss
    - shape_loss
    - motion_prior
    - motion_smooth
    - motion_consistency
  latitude:                  # latitude factor configurations
    scaling: 1.0             # when get the latitude, how much terrain point cloud to look at
  icp_loss:                  # icp loss between neighboring frames
    neighbor_num: 1          # nearest neighbor number
    least_pc: 10             # least LiDAR point number to consider  
    pc_limit: 1000           # maximum LiDAR point number, downsample the number of surpass the limit   
    box_scaling_prev: 1.1    # when segmenting the related point cloud, the scale in the previous box
    box_scaling_next: 1.5    # when segmenting the related point cloud, the scale in the next box    
    loss_type: L2            # loss type, such as L2, Huber
    ransac:                  # ransac options
      switch: false          # if use ransac
      num_iter: 200          # ransac iterations
      threshold: 0.2         # ransac outlier distance threshold
      ransac_limit: 10       # least inlier / LiDAR point number to consider
  shape_loss:                # shape loss configurations
    neighbor_num: 1          # nearest neighbor number
    least_pc: 10             # least LiDAR point number to consider  
    pc_limit: 1000           # maximum LiDAR point number, downsample the number of surpass the limit   
    box_scaling_next: 1.5    # when segmenting the related point cloud, the scale in the next box    
    loss_type: L2            # loss type, such as L2, Huber
    ransac:                  # ransac options
      switch: false          # if use ransac
      num_iter: 200          # ransac iterations
      threshold: 0.2         # ransac outlier distance threshold
      ransac_limit: 10       # least inlier / LiDAR point number to consider
  motion_prior: ~            # motion prior factor configurations
  motion_consistency: ~      # motion model loss configurations

weight:                      # loss weights
  latitude: 1.0
  motion_prior: 0.1
  motion_smooth: 0.0
  shape_loss: 1.0
  motion_consistency: 0.0
```

