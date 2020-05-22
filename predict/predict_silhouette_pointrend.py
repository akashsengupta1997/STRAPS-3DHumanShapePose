import os
import argparse
import numpy as np
import cv2

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, \
    DefaultPredictor

from PointRend.point_rend import add_pointrend_config


def setup_predictor():
    """
    Create configs and perform basic setups.
    """
    config_file = "PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml"
    weights = ['MODEL.WEIGHTS', 'PointRend/checkpoints/pointrend_rcnn_R50_fpn.pkl']
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(weights)
    predictor = DefaultPredictor(cfg)
    # cfg.freeze()
    # default_setup(cfg, args)
    return predictor


def get_largest_centred_mask(human_masks, orig_w, orig_h):
    """
    Args:
        human_masks: (N, img_wh, img_wh) human segmentation masks from a single image.
    Returns:
        Index of largest roughly-centred human mask.
    """
    mask_areas = np.sum(human_masks, axis=(1, 2))
    sorted_mask_indices = np.argsort(mask_areas)[::-1]  # Indices of masks sorted by area.
    mask_found = False
    i = 0
    while not mask_found and i < sorted_mask_indices.shape[0]:
        mask_index = sorted_mask_indices[i]
        mask = human_masks[mask_index, :, :]
        mask_pixels = np.argwhere(mask != 0)
        bbox_corners = np.amin(mask_pixels, axis=0), np.amax(mask_pixels, axis=0)  # (row_min, col_min), (row_max, col_max)
        bbox_centre = ((bbox_corners[0][0] + bbox_corners[1][0]) / 2.0,
                       (bbox_corners[0][1] + bbox_corners[1][1]) / 2.0)  # Centre in rows, columns (i.e. height, width)

        print(mask.shape, mask_pixels.shape, bbox_centre)
        if abs(bbox_centre[0] - orig_h / 2.0) < 120 and abs(bbox_centre[1] - orig_w / 2.0) < 70:
            largest_centred_mask_index = mask_index
            mask_found = True
        i += 1

    # If can't find mask sufficiently close to centre, just use biggest mask as prediction
    if not mask_found:
        largest_centred_mask_index = sorted_mask_indices[0]

    return largest_centred_mask_index


def predict_silhouette_pointrend(input_image):
    """

    """
    predictor = setup_predictor()

    orig_h, orig_w = input_image.shape[:2]
    outputs = predictor(input_image)['instances']
    print(outputs)
    classes = outputs.pred_classes
    masks = outputs.pred_masks
    human_masks = masks[classes == 0]
    human_masks = human_masks.cpu().detach().numpy()
    largest_centred_mask_index = get_largest_centred_mask(human_masks, orig_w, orig_h)
    human_mask = human_masks[largest_centred_mask_index, :, :].astype(np.uint8)
    print(human_mask.shape)
    overlay = cv2.addWeighted(input_image, 1.0,
                              255 * np.tile(human_mask[:, :, None], [1, 1, 3]),
                              0.5, gamma=0)

    return human_mask, overlay
