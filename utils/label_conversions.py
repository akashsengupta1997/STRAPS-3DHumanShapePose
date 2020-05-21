"""
Contains functions for label conversions.
"""
import numpy as np
import torch


def convert_densepose_to_6part_lsp_labels(densepose_seg):
    lsp_6part_seg = np.zeros_like(densepose_seg)

    lsp_6part_seg[densepose_seg == 1] = 6
    lsp_6part_seg[densepose_seg == 2] = 6
    lsp_6part_seg[densepose_seg == 3] = 2
    lsp_6part_seg[densepose_seg == 4] = 1
    lsp_6part_seg[densepose_seg == 5] = 4
    lsp_6part_seg[densepose_seg == 6] = 5
    lsp_6part_seg[densepose_seg == 7] = 5
    lsp_6part_seg[densepose_seg == 8] = 4
    lsp_6part_seg[densepose_seg == 9] = 5
    lsp_6part_seg[densepose_seg == 10] = 4
    lsp_6part_seg[densepose_seg == 11] = 5
    lsp_6part_seg[densepose_seg == 12] = 4
    lsp_6part_seg[densepose_seg == 13] = 5
    lsp_6part_seg[densepose_seg == 14] = 4
    lsp_6part_seg[densepose_seg == 15] = 1
    lsp_6part_seg[densepose_seg == 16] = 2
    lsp_6part_seg[densepose_seg == 17] = 1
    lsp_6part_seg[densepose_seg == 18] = 2
    lsp_6part_seg[densepose_seg == 19] = 1
    lsp_6part_seg[densepose_seg == 20] = 2
    lsp_6part_seg[densepose_seg == 21] = 1
    lsp_6part_seg[densepose_seg == 22] = 2
    lsp_6part_seg[densepose_seg == 23] = 3
    lsp_6part_seg[densepose_seg == 24] = 3

    return lsp_6part_seg


def convert_multiclass_to_binary_labels(multiclass_labels):
    """
    Converts multiclass segmentation labels into a binary mask.
    """
    binary_labels = np.zeros_like(multiclass_labels)
    binary_labels[multiclass_labels != 0] = 1

    return binary_labels

def convert_multiclass_to_binary_labels_torch(multiclass_labels):
    """
    Converts multiclass segmentation labels into a binary mask.
    """
    binary_labels = torch.zeros_like(multiclass_labels)
    binary_labels[multiclass_labels != 0] = 1

    return binary_labels


def convert_2Djoints_to_gaussian_heatmaps(joints2D, img_wh, std=4):
    """
    Converts 2D joints locations to img_wh x img_wh x num_joints gaussian heatmaps with given
    standard deviation var.
    """
    num_joints = joints2D.shape[0]
    size = 2 * std  # Truncate gaussian at 2 std from joint location.
    heatmaps = np.zeros((img_wh, img_wh, num_joints), dtype=np.float32)
    for i in range(joints2D.shape[0]):
        if np.all(joints2D[i] > -size) and np.all(joints2D[i] < img_wh-1+size):
            x, y = np.meshgrid(np.linspace(-size, size, 2*size),
                               np.linspace(-size, size, 2*size))
            d = np.sqrt(x * x + y * y)
            gaussian = np.exp(-(d ** 2 / (2.0 * std ** 2)))

            joint_centre = joints2D[i]
            hmap_start_x = max(0, joint_centre[0] - size)
            hmap_end_x = min(img_wh-1, joint_centre[0] + size)
            hmap_start_y = max(0, joint_centre[1] - size)
            hmap_end_y = min(img_wh-1, joint_centre[1] + size)

            g_start_x = max(0, size - joint_centre[0])
            g_end_x = min(2*size, 2*size - (size + joint_centre[0] - (img_wh-1)))
            g_start_y = max(0, size - joint_centre[1])
            g_end_y = min(2 * size, 2 * size - (size + joint_centre[1] - (img_wh-1)))

            heatmaps[hmap_start_y:hmap_end_y,
            hmap_start_x:hmap_end_x, i] = gaussian[g_start_y:g_end_y, g_start_x:g_end_x]

    return heatmaps


def convert_2Djoints_to_gaussian_heatmaps_torch(joints2D, img_wh, std=4):
    """
    Converts 2D joints locations to img_wh x img_wh x num_joints gaussian heatmaps with given
    standard deviation var.
    :param joints2D: (B, N, 2) tensor - batch of 2D joints.
    :return heatmaps: (B, N, img_wh, img_wh) - batch of 2D joint heatmaps.
    """
    joints2D_rounded = joints2D.int()
    batch_size = joints2D_rounded.shape[0]
    num_joints = joints2D_rounded.shape[1]
    device = joints2D_rounded.device
    heatmaps = torch.zeros((batch_size, num_joints, img_wh, img_wh), device=device).float()

    size = 2 * std  # Truncate gaussian at 2 std from joint location.
    x, y = torch.meshgrid(torch.linspace(-size, size, 2 * size),
                          torch.linspace(-size, size, 2 * size))
    x = x.to(device)
    y = y.to(device)
    d = torch.sqrt(x * x + y * y)
    gaussian = torch.exp(-(d ** 2 / (2.0 * std ** 2)))

    for i in range(batch_size):
        for j in range(num_joints):
            if torch.all(joints2D_rounded[i, j] > -size) and torch.all(joints2D_rounded[i, j] < img_wh-1+size):
                joint_centre = joints2D_rounded[i, j]
                hmap_start_x = max(0, joint_centre[0].item() - size)
                hmap_end_x = min(img_wh-1, joint_centre[0].item() + size)
                hmap_start_y = max(0, joint_centre[1].item() - size)
                hmap_end_y = min(img_wh-1, joint_centre[1].item() + size)

                g_start_x = max(0, size - joint_centre[0].item())
                g_end_x = min(2*size, 2*size - (size + joint_centre[0].item() - (img_wh-1)))
                g_start_y = max(0, size - joint_centre[1].item())
                g_end_y = min(2 * size, 2 * size - (size + joint_centre[1].item() - (img_wh-1)))

                heatmaps[i, j, hmap_start_y:hmap_end_y, hmap_start_x:hmap_end_x] = gaussian[g_start_y:g_end_y, g_start_x:g_end_x]

    return heatmaps

