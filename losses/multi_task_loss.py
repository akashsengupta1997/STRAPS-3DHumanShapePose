import torch
import torch.nn as nn
import numpy as np

import config

class HomoscedasticUncertaintyWeightedMultiTaskLoss(nn.Module):
    """
    Multi-task loss function. Loss weights are learnt via homoscedastic uncertainty (Kendall
    et al.) Losses can be applied on 3D vertices, 2D joints (projected), 3D joints, SMPL pose
    parameters (in the form of rotation matrices) and SMPL shape parameters.
    """
    def __init__(self,
                 losses_on,
                 init_loss_weights=None,
                 reduction='mean',
                 eps=1e-6):
        """
        :param losses_on: List of outputs to apply losses on.
        Subset of ['verts', 'joints2D', 'joints3D', 'pose_params', 'shape_params'].
        :param init_loss_weights: Initial multi-task loss weights.
        :param reduction: 'mean' or 'sum'
        :param eps: small constant
        """
        super(HomoscedasticUncertaintyWeightedMultiTaskLoss, self).__init__()

        self.losses_on = losses_on
        assert reduction in ['mean', 'sum'], "Invalid reduction for loss."

        if init_loss_weights is not None:
            # Initialise log variances using given init loss weights for vertices, joints2D,
            # joints3D, shape, pose
            init_verts_log_var = -np.log(init_loss_weights['verts'] + eps)
            init_joints2D_log_var = -np.log(init_loss_weights['joints2D'] + eps)
            init_joints3D_log_var = -np.log(init_loss_weights['joints3D'] + eps)
            init_pose_params_log_var = -np.log(init_loss_weights['pose_params'] + eps)
            init_shape_params_log_var = -np.log(init_loss_weights['shape_params'] + eps)
        else:
            # Initialise log variances to 0.
            init_verts_log_var = 0
            init_joints2D_log_var = 0
            init_joints3D_log_var = 0
            init_pose_params_log_var = 0
            init_shape_params_log_var = 0

        self.verts_log_var = nn.Parameter(torch.tensor(init_verts_log_var).float(),
                                          requires_grad=False)
        self.joints2D_log_var = nn.Parameter(torch.tensor(init_joints2D_log_var).float(),
                                             requires_grad=False)
        self.joints3D_log_var = nn.Parameter(torch.tensor(init_joints3D_log_var).float(),
                                             requires_grad=False)
        self.pose_params_log_var = nn.Parameter(torch.tensor(init_pose_params_log_var).float(),
                                                requires_grad=False)
        self.shape_params_log_var = nn.Parameter(torch.tensor(init_shape_params_log_var).float(),
                                                 requires_grad=False)

        if 'verts' in losses_on:
            self.verts_log_var.requires_grad = True
            self.verts_loss = nn.MSELoss(reduction=reduction)
        if 'joints2D' in losses_on:
            self.joints2D_log_var.requires_grad = True
            self.joints2D_loss = nn.MSELoss(reduction=reduction)
        if 'joints3D' in losses_on:
            self.joints3D_log_var.requires_grad = True
            self.joints3D_loss = nn.MSELoss(reduction=reduction)
        if 'shape_params' in losses_on:
            self.shape_params_log_var.requires_grad = True
            self.shape_params_loss = nn.MSELoss(reduction=reduction)
        if 'pose_params' in losses_on:
            self.pose_params_log_var.requires_grad = True
            self.pose_params_loss = nn.MSELoss(reduction=reduction)

    def forward(self, labels, outputs):

        total_loss = 0.
        loss_dict = {}

        if 'verts' in self.losses_on:
            verts_loss = self.verts_loss(outputs['verts'], labels['verts'])
            total_loss += verts_loss * torch.exp(-self.verts_log_var) + self.verts_log_var
            loss_dict['verts'] = verts_loss * torch.exp(-self.verts_log_var)

        if 'joints2D' in self.losses_on:
            joints2D_label = labels['joints2D']
            joints2D_pred = outputs['joints2D']

            if 'vis' in labels.keys():
                vis = labels['vis']  # joint visibility label - boolean
                joints2D_label = joints2D_label[vis, :]
                joints2D_pred = joints2D_pred[vis, :]

            joints2D_label = (2.0*joints2D_label) / config.REGRESSOR_IMG_WH - 1.0  # normalising j2d label
            joints2D_loss = self.joints2D_loss(joints2D_pred, joints2D_label)
            total_loss += joints2D_loss * torch.exp(-self.joints2D_log_var) + self.joints2D_log_var
            loss_dict['joints2D'] = joints2D_loss * torch.exp(-self.joints2D_log_var)

        if 'joints3D' in self.losses_on:
            joints3D_loss = self.joints3D_loss(outputs['joints3D'], labels['joints3D'])
            total_loss += joints3D_loss * torch.exp(-self.joints3D_log_var) + self.joints3D_log_var
            loss_dict['joints3D'] = joints3D_loss * torch.exp(-self.joints3D_log_var)

        if 'shape_params' in self.losses_on:
            shape_params_loss = self.shape_params_loss(outputs['shape_params'],
                                                       labels['shape_params'])
            total_loss += shape_params_loss * torch.exp(-self.shape_params_log_var) + self.shape_params_log_var
            loss_dict['shape_params'] = shape_params_loss * torch.exp(-self.shape_params_log_var)

        if 'pose_params' in self.losses_on:
            pose_params_loss = self.pose_params_loss(outputs['pose_params_rot_matrices'],
                                                     labels['pose_params_rot_matrices'])
            total_loss += pose_params_loss * torch.exp(-self.pose_params_log_var) + self.pose_params_log_var
            loss_dict['pose_params'] = pose_params_loss * torch.exp(-self.pose_params_log_var)

        if 'silhouette' in self.losses_on:
            silhouette_loss = self.silhouette_loss(outputs['silhouettes'], labels['silhouettes'])
            total_loss += silhouette_loss * torch.exp(-self.silhouette_log_var) + self.silhouette_log_var
            loss_dict['silhouette'] = silhouette_loss * torch.exp(-self.silhouette_log_var)

        return total_loss, loss_dict
