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
            l, h = delta_scale_range
            delta_scale = (h - l) * np.random.rand() + l
            scale_factor = orig_scale_factor + delta_scale
        else:
            scale_factor = orig_scale_factor

        if delta_centre_range is not None:
            l, h = delta_centre_range
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


def crop_and_resize_silhouette_joints(silhouette,
                                      joints2D,
                                      out_wh,
                                      image=None,
                                      image_out_wh=None,
                                      bbox_scale_factor=1.2):
    # Find bounding box around silhouette
    body_pixels = np.argwhere(silhouette != 0)
    bbox_centre, height, width = convert_bbox_corners_to_centre_hw(np.concatenate([np.amin(body_pixels, axis=0),
                                                                                   np.amax(body_pixels, axis=0)]))
    wh = max(height, width) * bbox_scale_factor  # Make bounding box square with sides = wh
    bbox_corners = convert_bbox_centre_hw_to_corners(bbox_centre, wh, wh)
    top_left = bbox_corners[:2].astype(np.int16)
    bottom_right = bbox_corners[2:].astype(np.int16)
    top_left_orig = top_left.copy()
    bottom_right_orig = bottom_right.copy()
    top_left[top_left < 0] = 0
    bottom_right[bottom_right < 0] = 0
    # Crop silhouette
    orig_height, orig_width = silhouette.shape[:2]
    silhouette = silhouette[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1]]
    # Pad silhouette if crop not square
    silhouette = cv2.copyMakeBorder(src=silhouette,
                                    top=max(0, -top_left_orig[0]),
                                    bottom=max(0, bottom_right_orig[0] - orig_height),
                                    left=max(0, -top_left_orig[1]),
                                    right=max(0, bottom_right_orig[1] - orig_width),
                                    borderType=cv2.BORDER_CONSTANT,
                                    value=0)
    crop_height, crop_width = silhouette.shape[:2]
    # Resize silhouette
    silhouette = cv2.resize(silhouette, (out_wh, out_wh),
                            interpolation=cv2.INTER_NEAREST)

    # Translate and resize joints2D
    joints2D = joints2D[:, :2] - top_left_orig[::-1]
    joints2D = joints2D * np.array([out_wh / float(crop_width),
                                    out_wh / float(crop_height)])

    if image is not None:
        # Crop image
        orig_height, orig_width = image.shape[:2]
        image = image[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1]]
        # Pad image if crop not square
        image = cv2.copyMakeBorder(src=image,
                                   top=max(0, -top_left_orig[0]),
                                   bottom=max(0, bottom_right_orig[0] - orig_height),
                                   left=max(0, -top_left_orig[1]),
                                   right=max(0, bottom_right_orig[1] - orig_width),
                                   borderType=cv2.BORDER_CONSTANT,
                                   value=0)
        # Resize silhouette
        image = cv2.resize(image, (image_out_wh, image_out_wh),
                           interpolation=cv2.INTER_LINEAR)

    return silhouette, joints2D, image

