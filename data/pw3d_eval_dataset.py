import cv2
import numpy as np
import os
import pickle
import torch
from torch.utils.data import Dataset

from utils.label_conversions import convert_multiclass_to_binary_labels, \
    convert_2Djoints_to_gaussian_heatmaps, convert_densepose_to_6part_lsp_labels


class PW3DEvalDataset(Dataset):
    def __init__(self, pw3d_dir_path, network_input, img_wh, num_classes,
                 silhouettes_from='densepose',
                 hmaps_gaussian_std=4):

        super(PW3DEvalDataset, self).__init__()

        # Paths
        densepose_masks_dir = os.path.join(pw3d_dir_path, 'densepose_masks')
        densepose_vis_dir = os.path.join(pw3d_dir_path, 'densepose_vis')
        pointrend_masks_dir = os.path.join(pw3d_dir_path, 'pointrend_R50FPN_masks')
        pointrend_vis_dir = os.path.join(pw3d_dir_path, 'pointrend_R50FPN_vis')
        kprcnn_keypoints_dir = os.path.join(pw3d_dir_path, 'keypoint_rcnn_results',
                                            'keypoints')
        kprcnn_keypoints_vis_dir = os.path.join(pw3d_dir_path, 'keypoint_rcnn_results',
                                                'keypoints_vis')

        # Data
        data = np.load(os.path.join(pw3d_dir_path, '3dpw_test.npz'))
        self.frame_fnames = data['imgname']
        self.pose = data['pose']
        self.shape = data['shape']
        self.gender = data['gender']

        self.network_input = network_input
        self.img_wh = img_wh
        self.num_classes = num_classes
        self.hmaps_gaussian_std = hmaps_gaussian_std

        assert silhouettes_from in ['densepose', 'pointrend']
        self.silhouettes_from = silhouettes_from

        self.densepose_masks_dir = densepose_masks_dir
        self.pointrend_masks_dir = pointrend_masks_dir
        self.kprcnn_keypoints_dir = kprcnn_keypoints_dir

        self.densepose_vis_dir = densepose_vis_dir
        self.pointrend_vis_dir = pointrend_vis_dir
        self.kprcnn_vis_dir = kprcnn_keypoints_vis_dir

    def __len__(self):
        return len(self.frame_fnames)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # Inputs
        fname = self.frame_fnames[index]
        densepose_mask_path = os.path.join(self.densepose_masks_dir, fname)
        pointrend_mask_path = os.path.join(self.pointrend_masks_dir, fname)
        kprcnn_keypoints_path = os.path.join(self.kprcnn_keypoints_dir,
                                             os.path.splitext(fname)[0] + '.npy')

        if self.network_input.startswith('seg'):
            densepose_mask = cv2.imread(densepose_mask_path, 0)
            img_in = convert_densepose_to_6part_lsp_labels(densepose_mask)
        elif self.network_input.startswith('silh'):
            if self.silhouettes_from == 'densepose' or not os.path.exists(pointrend_mask_path):
                densepose_mask = cv2.imread(densepose_mask_path, 0)
                img_in = convert_multiclass_to_binary_labels(densepose_mask)
            elif self.silhouettes_from == 'pointrend':
                img_in = cv2.imread(pointrend_mask_path, 0)
        orig_height, orig_width = img_in.shape[:2]
        img_in = cv2.resize(img_in, (self.img_wh, self.img_wh),
                            interpolation=cv2.INTER_NEAREST)
        img_in = img_in / float(self.num_classes)

        if self.network_input.endswith('joint_hmaps'):
            if os.path.exists(kprcnn_keypoints_path):
                kprcnn_keypoints = np.load(kprcnn_keypoints_path)
                kprcnn_keypoints = kprcnn_keypoints[:, :2]
                kprcnn_keypoints = kprcnn_keypoints * np.array(
                    [self.img_wh / float(orig_width),
                     self.img_wh / float(orig_height)])
                heatmaps = convert_2Djoints_to_gaussian_heatmaps(
                    kprcnn_keypoints.astype(np.int16), self.img_wh,
                    std=self.hmaps_gaussian_std)
            else:  # TODO this is pretty hacky - padding with zeros if no kprcnn results
                heatmaps = np.zeros((self.img_wh, self.img_wh, 17))
            input = np.concatenate([img_in[:, :, None], heatmaps], axis=-1)
            input = np.transpose(input, [2, 0, 1])
        else:
            input = img_in[None, :]

        # Targets
        pose = self.pose[index]
        shape = self.shape[index]
        gender = self.gender[index]

        input = torch.from_numpy(input).float()
        pose = torch.from_numpy(pose).float()
        shape = torch.from_numpy(shape).float()

        # Visualisation
        if self.silhouettes_from == 'densepose':
            densepose_vis_path = os.path.join(self.densepose_vis_dir, fname)
            vis_mask = cv2.cvtColor(cv2.imread(densepose_vis_path), cv2.COLOR_BGR2RGB)
        if self.silhouettes_from == 'pointrend':
            pointrend_vis_path = os.path.join(self.pointrend_vis_dir, fname)
            vis_mask = cv2.cvtColor(cv2.imread(pointrend_vis_path), cv2.COLOR_BGR2RGB)

        return {'input': input,
                'pose': pose,
                'shape': shape,
                'fname': fname,
                'gender': gender,
                'vis_mask': vis_mask}
