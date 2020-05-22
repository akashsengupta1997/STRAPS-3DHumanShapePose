import argparse
import glob
import os
import pickle
import sys
from typing import Any, ClassVar, Dict, List
import torch
import numpy as np

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.boxes import BoxMode
from detectron2.structures.instances import Instances

from DensePose.densepose import add_densepose_config
from DensePose.densepose.structures import DensePoseResult


_ACTION_REGISTRY: Dict[str, "Action"] = {}


def register_action(cls: type):
    """
    Decorator for action classes to automate action registration
    """
    global _ACTION_REGISTRY
    _ACTION_REGISTRY[cls.COMMAND] = cls
    return cls


def setup_config():
    config_file = "DensePose/configs/densepose_rcnn_R_101_FPN_s1x.yaml"
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = "DensePose/checkpoints/densepose_rcnn_R_101_fpn_s1x.pkl"
    cfg.freeze()
    return cfg


def get_largest_centred_bounding_box(bboxes, orig_w, orig_h):
    """
    Given an array of bounding boxes, return the index of the largest + roughly-centred
    bounding box.
    :param bboxes: (N, 4) array of [x1 y1 x2 y2] bounding boxes
    :param orig_w: original image width
    :param orig_h: original image height
    """
    bboxes_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    sorted_bbox_indices = np.argsort(bboxes_area)[::-1]  # Indices of bboxes sorted by area.
    bbox_found = False
    i = 0
    while not bbox_found and i < sorted_bbox_indices.shape[0]:
        bbox_index = sorted_bbox_indices[i]
        bbox = bboxes[bbox_index]
        bbox_centre = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)  # Centre (width, height)
        if abs(bbox_centre[0] - orig_w / 2.0) < orig_w/5.0 and abs(bbox_centre[1] - orig_h / 2.0) < orig_w/5.0:
            largest_centred_bbox_index = bbox_index
            bbox_found = True
        i += 1

    # If can't find bbox sufficiently close to centre, just use biggest bbox as prediction
    if not bbox_found:
        largest_centred_bbox_index = sorted_bbox_indices[0]

    return largest_centred_bbox_index


def predict_densepose(input_image):
    cfg = setup_config()
    predictor = DefaultPredictor(cfg)

    orig_h, orig_w = input_image.shape[:2]
    outputs = predictor(input_image)["instances"]
    print('DENSEPOSE')
    print(outputs)
    print(outputs.pred_boxes)
    print(outputs.pred_densepose)
    bboxes = outputs.pred_boxes.tensor.cpu().numpy()  # Multiple densepose predictions if there are multiple people in the image
    bboxes_XYWH = BoxMode.convert(bboxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    largest_centred_bbox_index = get_largest_centred_bounding_box(bboxes, orig_w, orig_h)  # Picks out centred person that is largest in the image.

    pred_densepose = outputs.pred_densepose.to_result(bboxes_XYWH)
    print(pred_densepose)
    # iuv_arr = DensePoseResult.decode_png_data(*densepose)

    # Round bbox to int
    # largest_bbox = bboxes[largest_centred_bbox_index]
    # w1 = largest_bbox[0]
    # w2 = largest_bbox[0] + iuv_arr.shape[2]
    # h1 = largest_bbox[1]
    # h2 = largest_bbox[1] + iuv_arr.shape[1]
    #
    # I_image = np.zeros((orig_h, orig_w))
    # I_image[int(h1):int(h2), int(w1):int(w2)] = iuv_arr[0, :, :]
    # U_image = np.zeros((orig_h, orig_w))
    # U_image[int(h1):int(h2), int(w1):int(w2)] = iuv_arr[1, :, :]
    # V_image = np.zeros((orig_h, orig_w))
    # V_image[int(h1):int(h2), int(w1):int(w2)] = iuv_arr[2, :, :]

    # Save visualisation and I image (i.e. segmentation mask)
    # vis_I_image = apply_colormap(I_image, vmin=0, vmax=24)
    # vis_I_image = vis_I_image[:, :, :3].astype(np.float32)
    # vis_I_image[I_image == 0, :] = np.zeros(3, dtype=np.float32)
    # overlay = cv2.addWeighted(frame,
    #                           0.6,
    #                           vis_I_image,
    #                           0.4,
    #                           gamma=0)

