import os
import cv2
import numpy as np
import torch
from smplx.lbs import batch_rodrigues

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

from PointRend.point_rend import add_pointrend_config
from DensePose.densepose import add_densepose_config

import config

from predict.predict_joints2D import predict_joints2D
from predict.predict_silhouette_pointrend import predict_silhouette_pointrend
from predict.predict_densepose import predict_densepose

from models.smpl_official import SMPL
from renderers.weak_perspective_pyrender_renderer import Renderer

from utils.image_utils import pad_to_square
from utils.label_conversions import convert_multiclass_to_binary_labels, \
    convert_2Djoints_to_gaussian_heatmaps
from utils.rigid_transform_utils import rot6d_to_rotmat

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def setup_detectron2_predictors(silhouettes_from='densepose'):
    # Keypoint-RCNN
    kprcnn_config_file = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    kprcnn_cfg = get_cfg()
    kprcnn_cfg.merge_from_file(model_zoo.get_config_file(kprcnn_config_file))
    kprcnn_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    kprcnn_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(kprcnn_config_file)
    kprcnn_cfg.freeze()
    joints2D_predictor = DefaultPredictor(kprcnn_cfg)

    if silhouettes_from == 'pointrend':
        # PointRend-RCNN-R50-FPN
        pointrend_config_file = "PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml"
        pointrend_cfg = get_cfg()
        add_pointrend_config(pointrend_cfg)
        pointrend_cfg.merge_from_file(pointrend_config_file)
        pointrend_cfg.MODEL.WEIGHTS = "checkpoints/pointrend_rcnn_R_50_fpn.pkl"
        pointrend_cfg.freeze()
        silhouette_predictor = DefaultPredictor(pointrend_cfg)
    elif silhouettes_from == 'densepose':
        # DensePose-RCNN-R101-FPN
        densepose_config_file = "DensePose/configs/densepose_rcnn_R_101_FPN_s1x.yaml"
        densepose_cfg = get_cfg()
        add_densepose_config(densepose_cfg)
        densepose_cfg.merge_from_file(densepose_config_file)
        densepose_cfg.MODEL.WEIGHTS = "checkpoints/densepose_rcnn_R_101_fpn_s1x.pkl"
        densepose_cfg.freeze()
        silhouette_predictor = DefaultPredictor(densepose_cfg)

    return joints2D_predictor, silhouette_predictor


def create_proxy_representation(silhouette,
                                joints2D,
                                in_wh,
                                out_wh):
    silhouette = cv2.resize(silhouette, (out_wh, out_wh),
                            interpolation=cv2.INTER_NEAREST)
    joints2D = joints2D[:, :2]
    joints2D = joints2D * np.array([out_wh / float(in_wh),
                                    out_wh / float(in_wh)])
    heatmaps = convert_2Djoints_to_gaussian_heatmaps(joints2D.astype(np.int16),
                                                     out_wh)
    proxy_rep = np.concatenate([silhouette[:, :, None], heatmaps], axis=-1)
    proxy_rep = np.transpose(proxy_rep, [2, 0, 1])  # (C, out_wh, out_WH)

    return proxy_rep


def predict_3D(input,
               regressor,
               device,
               silhouettes_from='densepose',
               proxy_rep_input_wh=512):

    # Set-up proxy representation predictors.
    joints2D_predictor, silhouette_predictor = setup_detectron2_predictors(silhouettes_from=silhouettes_from)

    # Set-up SMPL model.
    smpl = SMPL(config.SMPL_MODEL_DIR, batch_size=1).to(device)

    # Set-up renderer for visualisation.
    wp_renderer = Renderer(resolution=(proxy_rep_input_wh, proxy_rep_input_wh))

    if os.path.isdir(input):
        image_fnames = [f for f in sorted(os.listdir(input)) if f.endswith('.png') or
                        f.endswith('.jpg')]
        for fname in image_fnames:
            print("Predicting on:", fname)
            image = cv2.imread(os.path.join(input, fname))
            # Pre-process for 2D detectors
            image = pad_to_square(image)
            image = cv2.resize(image, (proxy_rep_input_wh, proxy_rep_input_wh),
                               interpolation=cv2.INTER_LINEAR)
            # Predict 2D
            joints2D, joints2D_vis = predict_joints2D(image, joints2D_predictor)
            if silhouettes_from == 'pointrend':
                silhouette, silhouette_vis = predict_silhouette_pointrend(image,
                                                                          silhouette_predictor)
            elif silhouettes_from == 'densepose':
                silhouette, silhouette_vis = predict_densepose(image, silhouette_predictor)
                silhouette = convert_multiclass_to_binary_labels(silhouette)

            # Create proxy representation
            proxy_rep = create_proxy_representation(silhouette, joints2D,
                                                    in_wh=proxy_rep_input_wh,
                                                    out_wh=config.REGRESSOR_IMG_WH)
            proxy_rep = proxy_rep[None, :, :, :]  # add batch dimension
            proxy_rep = torch.from_numpy(proxy_rep).float().to(device)

            # Predict 3D
            pred_cam_wp, pred_pose, pred_shape = regressor(proxy_rep)
            # Convert pred pose to rotation matrices
            if pred_pose.shape[-1] == 24 * 3:
                pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
            elif pred_pose.shape[-1] == 24 * 6:
                pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

            pred_smpl_output = smpl(body_pose=pred_pose_rotmats[:, 1:],
                                    global_orient=pred_pose_rotmats[:, 0].unsqueeze(1),
                                    betas=pred_shape,
                                    pose2rot=False)
            pred_vertices = pred_smpl_output.vertices

            pred_reposed_smpl_output = smpl(betas=pred_shape)
            pred_reposed_vertices = pred_reposed_smpl_output.vertices

            # Numpy-fying
            pred_vertices = pred_vertices.cpu().detach().numpy()[0]
            pred_reposed_vertices = pred_reposed_vertices.cpu().detach().numpy()[0]
            pred_cam_wp = pred_cam_wp.cpu().detach().numpy()[0]

            # TODO comment this out with newer models trained with translation before scaling
            pred_cam_wp[1:] = pred_cam_wp[1:] / pred_cam_wp[0]  # Need to do this division because of different cam translation convention (described in my research diary)
            rend_img = wp_renderer.render(verts=pred_vertices, cam=pred_cam_wp, img=image)
            rend_reposed_img = wp_renderer.render(verts=pred_reposed_vertices,
                                                  cam=np.array([0., 0., proxy_rep_input_wh/10]))

            plt.figure()
            plt.subplot(231)
            plt.imshow(joints2D_vis)
            plt.subplot(232)
            plt.imshow(silhouette)
            plt.subplot(233)
            plt.imshow(silhouette_vis)
            plt.subplot(234)
            plt.imshow(np.sum(proxy_rep[0], axis=0))
            plt.subplot(235)
            plt.imshow(rend_img)
            plt.subplot(236)
            plt.imshow(rend_reposed_img)
            plt.show()