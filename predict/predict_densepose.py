import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.boxes import BoxMode

from DensePose.densepose import add_densepose_config
from DensePose.densepose.structures import DensePoseResult



def apply_colormap(image, vmin=None, vmax=None, cmap='viridis', cmap_seed=1):
    """
    Apply a matplotlib colormap to an image.

    This method will preserve the exact image size. `cmap` can be either a
    matplotlib colormap name, a discrete number, or a colormap instance. If it
    is a number, a discrete colormap will be generated based on the HSV
    colorspace. The permutation of colors is random and can be controlled with
    the `cmap_seed`. The state of the RNG is preserved.
    """
    image = image.astype("float64")  # Returns a copy.
    # Normalization.
    if vmin is not None:
        imin = float(vmin)
        image = np.clip(image, vmin, sys.float_info.max)
    else:
        imin = np.min(image)
    if vmax is not None:
        imax = float(vmax)
        image = np.clip(image, -sys.float_info.max, vmax)
    else:
        imax = np.max(image)
    image -= imin
    image /= (imax - imin)
    # Visualization.
    cmap_ = plt.get_cmap(cmap)
    vis = cmap_(image, bytes=True)
    return vis


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
    bboxes = outputs.pred_boxes.tensor.cpu()  # Multiple densepose predictions if there are multiple people in the image
    bboxes_XYWH = BoxMode.convert(bboxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    bboxes = bboxes.cpu().detach().numpy()
    largest_centred_bbox_index = get_largest_centred_bounding_box(bboxes, orig_w, orig_h)  # Picks out centred person that is largest in the image.

    pred_densepose = outputs.pred_densepose.to_result(bboxes_XYWH)
    iuv_arr = DensePoseResult.decode_png_data(*pred_densepose.results[largest_centred_bbox_index])

    # Round bbox to int
    largest_bbox = bboxes[largest_centred_bbox_index]
    w1 = largest_bbox[0]
    w2 = largest_bbox[0] + iuv_arr.shape[2]
    h1 = largest_bbox[1]
    h2 = largest_bbox[1] + iuv_arr.shape[1]

    I_image = np.zeros((orig_h, orig_w))
    I_image[int(h1):int(h2), int(w1):int(w2)] = iuv_arr[0, :, :]
    # U_image = np.zeros((orig_h, orig_w))
    # U_image[int(h1):int(h2), int(w1):int(w2)] = iuv_arr[1, :, :]
    # V_image = np.zeros((orig_h, orig_w))
    # V_image[int(h1):int(h2), int(w1):int(w2)] = iuv_arr[2, :, :]

    vis_I_image = apply_colormap(I_image, vmin=0, vmax=24)
    print(vis_I_image.dtype)
    vis_I_image = vis_I_image[:, :, :3].astype(np.float32)
    vis_I_image[I_image == 0, :] = np.zeros(3, dtype=np.float32)
    overlay_vis = cv2.addWeighted(input_image.astype(np.float32),
                                  0.6,
                                  vis_I_image,
                                  0.4,
                                  gamma=0)
    print('DP', input_image.max(), vis_I_image.max(), input_image.min(), vis_I_image.min(), overlay_vis.max(), overlay_vis.min())
    return I_image, overlay_vis

