import torch
import numpy as np


def random_joints2D_deviation(joints2D,
                              delta_j2d_dev_range=[-5, 5],
                              delta_j2d_hip_dev_range=[-15, 15]):
    """
    Deviate 2D joint locations with uniform random noise.
    :param joints2D: (bs, num joints, num joints)
    :param delta_j2d_dev_range: uniform noise range.
    :param delta_j2d_hip_dev_range: uniform noise range for hip joints. You may wish to make
    this bigger than for other joints since hip joints are semantically hard to localise and
    can be predicted inaccurately by joint detectors.
    """
    hip_joints = [11, 12]
    other_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14,15, 16]
    batch_size = joints2D.shape[0]
    device = joints2D.device

    h, l = delta_j2d_dev_range
    delta_j2d_dev = (h - l) * torch.rand(batch_size, len(other_joints), 2, device=device) + l
    joints2D[:, other_joints, :] = joints2D[:, other_joints, :] + delta_j2d_dev

    h, l = delta_j2d_hip_dev_range
    delta_j2d_hip_dev_range = (h - l) * torch.rand(batch_size, len(hip_joints), 2, device=device) + l
    joints2D[:, hip_joints, :] = joints2D[:, hip_joints, :] + delta_j2d_hip_dev_range

    return joints2D


def random_remove_bodyparts(seg, classes_to_remove, probabilities_to_remove):
    """
    Randomly remove bodyparts from silhouette/segmentation (i.e. set pixels to background
    class).
    :param seg: (bs, wh, wh)
    :param classes_to_remove: list of classes to remove. Classes are integers (as defined in
    nmr_renderer.py).
    :param probabilities_to_remove: probability of removal for each class.
    """
    assert len(classes_to_remove) == len(probabilities_to_remove)

    batch_size = seg.shape[0]
    for i in range(len(classes_to_remove)):
        class_to_remove = classes_to_remove[i]
        prob_to_remove = probabilities_to_remove[i]

        # Determine which samples to augment in the batch
        rand_vec = np.random.rand(batch_size) < prob_to_remove
        samples_to_augment = seg[rand_vec].clone()

        samples_to_augment[samples_to_augment == class_to_remove] = 0
        seg[rand_vec] = samples_to_augment

    return seg


def random_occlude(seg, occlude_probability=0.5, occlude_box_dim=48):
    """
    Randomly occlude silhouette/part segmentation with boxes.
    :param seg: (bs, wh, wh)
    """
    batch_size = seg.shape[0]
    seg_wh = seg.shape[-1]
    seg_centre = seg_wh/2
    x_h, x_l = seg_centre - 0.3*seg_wh/2, seg_centre + 0.3*seg_wh/2
    y_h, y_l = seg_centre - 0.3*seg_wh/2, seg_centre + 0.3*seg_wh/2

    x = (x_h - x_l) * np.random.rand(batch_size) + x_l
    y = (y_h - y_l) * np.random.rand(batch_size) + y_l
    box_x1 = (x - occlude_box_dim / 2).astype(np.int16)
    box_x2 = (x + occlude_box_dim / 2).astype(np.int16)
    box_y1 = (y - occlude_box_dim / 2).astype(np.int16)
    box_y2 = (y + occlude_box_dim / 2).astype(np.int16)

    rand_vec = np.random.rand(batch_size)
    for i in range(batch_size):
        if rand_vec[i] < occlude_probability:
            seg[i, box_x1[i]:box_x2[i], box_y1[i]:box_y2[i]] = 0

    return seg


def augment_proxy_representation(orig_segs, orig_joints2D,
                                 remove_appendages, deviate_joints2D, occlude_seg,
                                 remove_appendages_classes, remove_appendages_probabilities,
                                 delta_j2d_dev_range, delta_j2d_hip_dev_range,
                                 occlude_probability, occlude_box_dim):
    new_segs = orig_segs.clone()
    new_joints2D = orig_joints2D.clone()

    if remove_appendages:
        new_segs = random_remove_bodyparts(new_segs,
                                           classes_to_remove=remove_appendages_classes,
                                           probabilities_to_remove=remove_appendages_probabilities)
    if occlude_seg:
        new_segs = random_occlude(new_segs,
                                  occlude_probability=occlude_probability,
                                  occlude_box_dim=occlude_box_dim)

    if deviate_joints2D:
        new_joints2D = random_joints2D_deviation(new_joints2D,
                                                 delta_j2d_dev_range=delta_j2d_dev_range,
                                                 delta_j2d_hip_dev_range=delta_j2d_hip_dev_range)

    return new_segs, new_joints2D


def random_verts2D_deviation(vertices, delta_verts2d_dev_range=[-0.01, 0.01]):
    """
    Randomly add 2D uniform noise to vertices to create silhouettes/part segmentations with
    corrupted edges.
    :param vertices: (bs, 6890, 3)
    :param delta_verts2d_dev_range: range of uniform noise.
    """
    batch_size = vertices.shape[0]
    num_verts = vertices.shape[1]
    device = vertices.device

    noisy_vertices = vertices.clone()

    h, l = delta_verts2d_dev_range
    delta_verts2d_dev = (h - l) * torch.rand(batch_size, num_verts, 2, device=device) + l
    noisy_vertices[:, :, :2] = noisy_vertices[:, :, :2] + delta_verts2d_dev

    return noisy_vertices


