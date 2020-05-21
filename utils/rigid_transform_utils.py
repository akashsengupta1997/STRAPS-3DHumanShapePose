import cv2
import torch
import numpy as np
from torch.nn import functional as F


def rotate_translate_verts_torch(vertices, axis, angle, trans):
    """
    Rotates and translates batch of vertices.
    :param vertices: B, N, 3
    :param axis: 3,
    :param angle: angle in radians
    :param trans: 3,
    :return:
    """
    r = angle * axis
    R = cv2.Rodrigues(r)[0]
    R = torch.from_numpy(R.astype(np.float32)).to(vertices.device)
    trans = torch.from_numpy(trans.astype(np.float32)).to(vertices.device)

    vertices = torch.einsum('ij,bkj->bki', R, vertices)
    vertices = vertices + trans

    return vertices


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)  # Ensuring columns are unit vectors
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)  # Ensuring column 1 and column 2 are orthogonal
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)