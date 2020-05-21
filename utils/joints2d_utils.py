import numpy as np
import torch


def undo_keypoint_normalisation(normalised_keypoints, img_wh):
    """
    Converts normalised keypoints from [-1, 1] space to pixel space i.e. [0, img_wh]
    """
    keypoints = (normalised_keypoints + 1) * (img_wh/2.0)
    return keypoints


def check_joints2d_visibility(joints2d, img_wh):
    vis = np.ones(joints2d.shape[1])
    vis[joints2d[0] > img_wh] = 0
    vis[joints2d[1] > img_wh] = 0
    vis[joints2d[0] < 0] = 0
    vis[joints2d[1] < 0] = 0

    return vis


def check_joints2d_visibility_torch(joints2d, img_wh):
    """
    Checks if 2D joints are within the image dimensions.
    """
    vis = torch.ones(joints2d.shape[:2], device=joints2d.device, dtype=torch.bool)
    vis[joints2d[:, :, 0] > img_wh] = 0
    vis[joints2d[:, :, 1] > img_wh] = 0
    vis[joints2d[:, :, 0] < 0] = 0
    vis[joints2d[:, :, 1] < 0] = 0

    return vis