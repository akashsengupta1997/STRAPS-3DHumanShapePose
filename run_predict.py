import os
import cv2

from predict.predict_joints2D import predict_joints2D
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# TODO pre-processing step (pad to square)
# TODO pointrend predict, DP predict, 3D predict
def predict_on_folder(input_folder):
    image_fnames = [f for f in sorted(os.listdir(input_folder)) if f.endswith('.png') or
                    f.endswith('.jpg')]
    for fname in image_fnames:
        print(fname)
        image = cv2.imread(os.path.join(input_folder, fname))
        keypoints, joints2D_vis = predict_joints2D(image)
        print(keypoints.shape)
        plt.imshow(joints2D_vis)
        plt.show()


predict_on_folder("/data/cvfs/as2562/datasets/toy_datasets/testing/cropped_frames")