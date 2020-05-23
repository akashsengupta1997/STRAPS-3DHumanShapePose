import os
import cv2
import numpy as np


def pad_to_square(image):
    """
    Pad image to square shape.
    """
    height, width = image.shape[:2]

    if width < height:
        border_width = (height - width) // 2
        image = cv2.copyMakeBorder(image, 0, 0, border_width, border_width,
                                   cv2.BORDER_CONSTANT, value=0)
    else:
        border_width = (width - height) // 2
        image = cv2.copyMakeBorder(image, border_width, border_width, 0, 0,
                                   cv2.BORDER_CONSTANT, value=0)

    return image

def convert_bbox_corners_to_centre_hw(bbox_corners):
    """
    Converst bbox coordinates from x1, y1, x2, y2 to centre, height, width.
    """
    x1, y1, x2, y2 = bbox_corners
    centre = np.array([(x1+x2)/2.0, (y1+y2)/2.0])
    height = x2 - x1
    width = y2 - y1

    return centre, height, width


def convert_bbox_centre_hw_to_corners(centre, height, width):
    x1 = centre[0] - height/2.0
    x2 = centre[0] + height/2.0
    y1 = centre[1] - width/2.0
    y2 = centre[1] + width/2.0

    return np.array([x1, y1, x2, y2])


def batch_crop_seg_to_bounding_box(seg, joints2D, orig_scale_factor=1.2, delta_scale_range=None, delta_centre_range=None):
    """
    seg: (bs, wh, wh)
    joints2D: (bs, num joints, 2)
    scale: bbox expansion scale
    """
    all_cropped_segs = []
    all_cropped_joints2D = []
    for i in range(seg.shape[0]):
        body_pixels = np.argwhere(seg[i] != 0)
        bbox_corners = np.amin(body_pixels, axis=0), np.amax(body_pixels, axis=0)
        bbox_corners = np.concatenate(bbox_corners)
        centre, height, width = convert_bbox_corners_to_centre_hw(bbox_corners)
        if delta_scale_range is not None:
            h, l = delta_scale_range
            delta_scale = (h - l) * np.random.rand() + l
            scale_factor = orig_scale_factor + delta_scale
        else:
            scale_factor = orig_scale_factor

        if delta_centre_range is not None:
            h, l = delta_centre_range
            delta_centre = (h - l) * np.random.rand(2) + l
            centre = centre + delta_centre

        wh = max(height, width) * scale_factor

        bbox_corners = convert_bbox_centre_hw_to_corners(centre, wh, wh)

        top_left = bbox_corners[:2].astype(np.int16)
        bottom_right = bbox_corners[2:].astype(np.int16)
        top_left[top_left < 0] = 0
        bottom_right[bottom_right < 0] = 0

        cropped_joints2d = joints2D[i] - top_left[::-1]
        cropped_seg = seg[i, top_left[0]: bottom_right[0], top_left[1]: bottom_right[1]]
        all_cropped_joints2D.append(cropped_joints2d)
        all_cropped_segs.append(cropped_seg)
    return all_cropped_segs, all_cropped_joints2D


def batch_resize(all_cropped_segs, all_cropped_joints2D, img_wh):
    """
    all_cropped_seg: list of cropped segs with len = batch size
    """
    all_resized_segs = []
    all_resized_joints2D = []
    for i in range(len(all_cropped_segs)):
        seg = all_cropped_segs[i]
        orig_height, orig_width = seg.shape[:2]
        resized_seg = cv2.resize(seg, (img_wh, img_wh), interpolation=cv2.INTER_NEAREST)
        all_resized_segs.append(resized_seg)

        joints2D = all_cropped_joints2D[i]
        resized_joints2D = joints2D * np.array([img_wh / float(orig_width),
                                        img_wh / float(orig_height)])
        all_resized_joints2D.append(resized_joints2D)

    all_resized_segs = np.stack(all_resized_segs, axis=0)
    all_resized_joints2D = np.stack(all_resized_joints2D, axis=0)

    return all_resized_segs, all_resized_joints2D

