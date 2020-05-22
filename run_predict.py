import os
import cv2

from predict.predict_joints2D import predict_joints2D
from predict.predict_silhouette_pointrend import predict_silhouette_pointrend
from predict.predict_densepose import predict_densepose
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# TODO pre-processing step (pad to square)
# TODO load all predictor only one time (in this script somewhere)
# TODO DP predict, 3D predict
def predict_on_folder(input_folder):
    image_fnames = [f for f in sorted(os.listdir(input_folder)) if f.endswith('.png') or
                    f.endswith('.jpg')]
    for fname in image_fnames:
        print("Predicting on:", fname)
        image = cv2.imread(os.path.join(input_folder, fname))
        keypoints, joints2D_vis = predict_joints2D(image)
        point_rend_mask, point_rend_vis = predict_silhouette_pointrend(image)
        densepose_seg, densepose_vis = predict_densepose(image)
        plt.figure()
        plt.subplot(231)
        plt.imshow(joints2D_vis)
        plt.subplot(232)
        plt.imshow(point_rend_mask)
        plt.subplot(233)
        plt.imshow(point_rend_vis)
        plt.subplot(234)
        plt.imshow(densepose_seg)
        plt.subplot(235)
        plt.imshow(densepose_vis)
        plt.show()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
predict_on_folder("/data/cvfs/as2562/datasets/toy_datasets/testing/cropped_frames")