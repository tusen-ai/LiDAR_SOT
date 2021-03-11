""" Reject the outlier point pairs in computation of icp loss
    The main function is the ransac
"""
import numpy as np, random
from sot_3d.utils.geometry import make_transformation_matrix


def ransac(src, tgt, min_point_num, num_iter, threshold, reference_motion, confidence=0.99):
    """ ransac interface. For input point clouds in corresponding pairs (src, tgt),
        compute the inlier indices.
        reference_motion also provides the z information.
    Args:
        src: source point clouds, paired with tgt
        tgt: target point clouds, paired with src
        min_point_num: minimum point cloud number
        num_iter: ransac parameter
        threshold: ransac parameter, distance to tolerate
        reference_motion: an initialized reference motion
        confidence: ransac parameter. defaults to 0.99.
    Return:
        indices for inlier pairs
    """
    n_iters = num_iter
    pc_num = tgt.shape[0]
    max_inlr, transform_ref = 0, None
    mask = np.zeros(pc_num, dtype=np.int32)

    # initialize ransac with reference motion
    tf_ref = make_transformation_matrix(reference_motion)
    dist = calc_dist(src, tgt, tf_ref)
    tmp_mask = (dist <= threshold)
    tgt_ref = tgt[tmp_mask, :]
    src_ref = src[tmp_mask, :]
    max_inlr = np.sum(tmp_mask)

    iter_index = 0
    while iter_index < n_iters:
        idx = np.asarray(random.sample(range(pc_num), min_point_num))
        src_sample = src[idx, :]
        tgt_sample = tgt[idx, :]

        transform = estimate_rigid_transform(src=src_sample, tgt=tgt_sample, 
            reference_motion=reference_motion)
        dist = calc_dist(src, tgt, transform)
        tmp_mask = (dist <= threshold)
        num_inlr = np.sum(tmp_mask)

        if num_inlr > max_inlr:
            max_inlr = num_inlr
            if num_inlr > min_point_num:
                src_ref = src[tmp_mask, :]
                tgt_ref = tgt[tmp_mask, :]
            outlier_ratio = (pc_num - num_inlr) / pc_num
            n_iters = update_num_iters(confidence, outlier_ratio, min_point_num, n_iters)
        iter_index += 1
    
    tf_ref = estimate_rigid_transform(src=src_ref, tgt=tgt_ref, reference_motion=reference_motion)
    dist_ref = calc_dist(src, tgt, tf_ref)
    mask = (dist_ref <= threshold)
    return mask


def calc_dist(src, tgt, trans):
    """ point level distance after transformation
    """
    new_points = np.concatenate((src,
                                 np.ones(src.shape[0])[:, np.newaxis]),
                                 axis=1)
    new_points = trans @ new_points.T
    new_points = new_points.T[:, :3]

    dist = tgt[:, :2] - new_points[:, :2]
    dist = np.linalg.norm(dist, axis=1)
    return dist


def update_num_iters(p, ep, modelPoints, maxIters):
    p = max(p, 0)
    p = min(p, 1)
    ep = max(ep, 0)
    ep = min(ep, 1)

    num = np.log(1 - p + 1e-6)
    denom = np.log(1 - (1 - ep) ** modelPoints + 1e-6)

    return maxIters if denom >= 0 or -num >= maxIters * (-denom) else np.round(num / (denom + 1e-6))


def estimate_rigid_transform(src, tgt, reference_motion):
    N = tgt.shape[0]
    # we need at least 3 points
    if N < 3:
        print("Not enough points to estimate transform. At least 3 points needed")

    pts1 = tgt[:, :2]
    pts2 = src[:, :2]
    # keep going with the real algorithm, follow the paper
    d = pts1.T
    m = pts2.T

    cd = np.mean(d, axis=1)
    cm = np.mean(m, axis=1)

    # centered (set centroid to 0)
    d_c = (d.T - cd).T
    m_c = (m.T - cm).T

    H = np.dot(m_c, d_c.T)
    U, S, V = np.linalg.svd(H)  # such that H = U * S * V, and NOT H = u * S * V.T

    R = np.dot(V.T, U.T)
    t = cd - np.dot(R, cm)

    # build the transform matrix, such that [d 1] = TF * [m 1]
    transform = np.eye(4)
    transform[0:2, 0:2] = R
    transform[0:2, 3] = t
    transform[2, 3] = reference_motion[2]
    return transform