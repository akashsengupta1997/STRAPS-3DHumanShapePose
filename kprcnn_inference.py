import cv2
import os
import argparse
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


# predictor returns bboxes, classes and pred keypoints
# pred keypoints has shape (N, K, 3): N = num instances, K = num keypoints (17), the 3 is
# (x, y, visibility/score/confidence)


def get_largest_centred_bounding_box(bboxes, orig_w, orig_h):
    """
    Args:
        bboxes: (N, 4) array of x1 y1 x2 y2 bounding boxes.

    Returns:
        Index of largest and roughtly centred bounding box.
    """
    bboxes_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    sorted_bbox_indices = np.argsort(bboxes_area)[::-1]  # Indices of bboxes sorted by area.
    bbox_found = False
    i = 0
    while not bbox_found and i < sorted_bbox_indices.shape[0]:
        bbox_index = sorted_bbox_indices[i]
        bbox = bboxes[bbox_index]
        bbox_centre = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)  # Centre in width, height
        if abs(bbox_centre[0] - orig_w / 2.0) < 80 and abs(bbox_centre[1] - orig_h / 2.0) < 80:
            largest_centred_bbox_index = bbox_index
            bbox_found = True
        i += 1

    # If can't find bbox sufficiently close to centre, just use biggest bbox as prediction
    if not bbox_found:
        largest_centred_bbox_index = sorted_bbox_indices[0]

    return largest_centred_bbox_index


def predict_on_folder(in_folder, out_folder, config_file, save_kps_in_separate_files=False):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(config_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
    predictor = DefaultPredictor(cfg)

    os.makedirs(os.path.join(out_folder, 'keypoints_vis'), exist_ok=True)
    if save_kps_in_separate_files:
        os.makedirs(os.path.join(out_folder, 'keypoints'), exist_ok=True)

    image_fnames = [f for f in sorted(os.listdir(in_folder)) if f.endswith('.png')]
    all_keypoints = []
    for fname in image_fnames:
        print(fname)
        image = cv2.imread(os.path.join(in_folder, fname))
        orig_h, orig_w = image.shape[:2]
        outputs = predictor(image)
        bboxes = outputs['instances'].pred_boxes.tensor.cpu().numpy()
        print(bboxes.shape, bboxes)
        if bboxes.shape[0] == 0:  # Can't find any people in image
            keypoints = np.zeros((17, 3))
            all_keypoints.append(keypoints)
        else:
            largest_bbox_index = get_largest_centred_bounding_box(bboxes, orig_w, orig_h)
            keypoints = outputs['instances'].pred_keypoints.cpu().numpy()
            keypoints = keypoints[largest_bbox_index]

            if not save_kps_in_separate_files:
                all_keypoints.append(keypoints)
            else:
                print(keypoints.shape)
                save_keypoints_path = os.path.join(out_folder, 'keypoints', os.path.splitext(fname)[0] + '.npy')
                np.save(save_keypoints_path, keypoints)

            for j in range(keypoints.shape[0]):
                cv2.circle(image, (keypoints[j, 0], keypoints[j, 1]), 5, (0, 255, 0), -1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.5
                fontColor = (0, 0, 255)
                cv2.putText(image, str(j), (keypoints[j, 0], keypoints[j, 1]),
                            font, fontScale, fontColor, lineType=2)
            save_vis_path = os.path.join(out_folder, 'keypoints_vis', fname)
            cv2.imwrite(save_vis_path, image)

    if not save_kps_in_separate_files:
        all_keypoints = np.stack(all_keypoints, axis=0)
        print(all_keypoints.shape)
        save_keypoints_path = os.path.join(out_folder, 'keypoints.npy')
        np.save(save_keypoints_path, all_keypoints)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_folder', type=str)
    parser.add_argument('--config_file', type=str,
                        help='config file name relative to detectron2 configs directory')
    parser.add_argument('--out_folder', type=str)
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--save_kps_separate', action='store_true')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.out_folder == 'dataset':
        out_folder = args.in_folder.replace('cropped_frames', 'keypoint_rcnn_results')
    elif args.out_folder == 'h36m':
        out_folder = args.in_folder.replace('cropped_frames', 'keypoint_rcnn_results')
    else:
        out_folder = args.out_folder

    predict_on_folder(args.in_folder, out_folder, args.config_file, save_kps_in_separate_files=args.save_kps_separate)
