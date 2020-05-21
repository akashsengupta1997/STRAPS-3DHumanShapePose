import numpy as np
import os

from utils.eval_utils import procrustes_analysis_batch, scale_and_translation_transform_batch
from utils.joints2d_utils import undo_keypoint_normalisation


class EvalMetricsTracker:
    """
    Tracks metrics during evaluation.
    """
    def __init__(self, metrics_to_track, img_wh=None, save_path=None,
                 save_per_frame_metrics=False):

        self.metrics_to_track = metrics_to_track
        self.img_wh = img_wh

        self.metric_sums = None
        self.total_samples = 0
        self.save_per_frame_metrics = save_per_frame_metrics
        self.save_path = save_path
        print('\nInitialised metrics tracker.')

    def initialise_metric_sums(self):
        self.metric_sums = {}
        for metric_type in self.metrics_to_track:
            if metric_type == 'silhouette_ious':
                self.metric_sums['num_true_positives'] = 0.
                self.metric_sums['num_false_positives'] = 0.
                self.metric_sums['num_true_negatives'] = 0.
                self.metric_sums['num_false_negatives'] = 0.
            else:
                self.metric_sums[metric_type] = 0.

    def initialise_per_frame_metric_lists(self):
        self.per_frame_metrics = {}
        for metric_type in self.metrics_to_track:
            self.per_frame_metrics[metric_type] = []

    def update_per_batch(self, pred_dict, target_dict, num_input_samples,
                         return_transformed_points=False):
        self.total_samples += num_input_samples
        if return_transformed_points:
            return_dict = {}
        # -------- Update metrics sums --------
        if 'pves' in self.metrics_to_track:
            pve_batch = np.linalg.norm(pred_dict['verts'] - target_dict['verts'],
                                       axis=-1)  # (bsize, 6890)
            self.metric_sums['pves'] += np.sum(pve_batch)  # scalar
            self.per_frame_metrics['pves'].append(np.mean(pve_batch, axis=-1))  # (bs,)

        # Scale and translation correction
        if 'pves_sc' in self.metrics_to_track:
            pred_vertices = pred_dict['verts']  # (bsize, 6890, 3)
            target_vertices = target_dict['verts']  # (bsize, 6890, 3)
            pred_vertices_sc = scale_and_translation_transform_batch(pred_vertices,
                                                                     target_vertices)
            pve_sc_batch = np.linalg.norm(pred_vertices_sc - target_vertices,
                                          axis=-1)  # (bs, 6890)
            self.metric_sums['pves_sc'] += np.sum(pve_sc_batch)  # scalar
            self.per_frame_metrics['pves_sc'].append(np.mean(pve_sc_batch, axis=-1))  # (bs,)
            if return_transformed_points:
                return_dict['pred_vertices_sc'] = pred_vertices_sc

        # Procrustes analysis
        if 'pves_pa' in self.metrics_to_track:
            pred_vertices = pred_dict['verts']  # (bsize, 6890, 3)
            target_vertices = target_dict['verts']  # (bsize, 6890, 3)
            pred_vertices_pa = procrustes_analysis_batch(pred_vertices, target_vertices)
            pve_pa_batch = np.linalg.norm(pred_vertices_pa - target_vertices,
                                          axis=-1)  # (bsize, 6890)
            self.metric_sums['pves_pa'] += np.sum(pve_pa_batch)  # scalar
            self.per_frame_metrics['pves_pa'].append(np.mean(pve_pa_batch, axis=-1))  # (bs,)
            if return_transformed_points:
                return_dict['pred_vertices_pa'] = pred_vertices_pa

        # Reposed
        if 'pve-ts' in self.metrics_to_track:
            pvet_batch = np.linalg.norm(pred_dict['reposed_verts'] - target_dict['reposed_verts'],
                                        axis=-1)
            self.metric_sums['pve-ts'] += np.sum(pvet_batch)
            self.per_frame_metrics['pve-ts'].append(np.mean(pvet_batch, axis=-1))  # (bs,)

        # Reposed + Scale and translation correction
        if 'pve-ts_sc' in self.metrics_to_track:
            pred_reposed_vertices = pred_dict['reposed_verts']
            target_reposed_vertices = target_dict['reposed_verts']
            pred_reposed_vertices_sc = scale_and_translation_transform_batch(pred_reposed_vertices,
                                                                             target_reposed_vertices)
            pvet_sc_batch = np.linalg.norm(pred_reposed_vertices_sc - target_reposed_vertices,
                                           axis=-1)  # (bs, 6890)
            self.metric_sums['pve-ts_sc'] += np.sum(pvet_sc_batch)  # scalar
            self.per_frame_metrics['pve-ts_sc'].append(np.mean(pvet_sc_batch, axis=-1))  # (bs,)
            if return_transformed_points:
                return_dict['pred_reposed_vertices_sc'] = pred_reposed_vertices_sc

        # Reposed + Procrustes analysis - this doesn't make practical sense for reposed.
        if 'pve-ts_pa' in self.metrics_to_track:
            pred_reposed_vertices = pred_dict['reposed_verts']
            target_reposed_vertices = target_dict['reposed_verts']
            pred_reposed_vertices_pa = procrustes_analysis_batch(pred_reposed_vertices,
                                                                 target_reposed_vertices)
            pvet_pa_batch = np.linalg.norm(pred_reposed_vertices_pa - target_reposed_vertices,
                                           axis=-1)  # (bsize, 6890)
            self.metric_sums['pve-ts_pa'] += np.sum(pvet_pa_batch)  # scalar
            self.per_frame_metrics['pve_ts_pa'].append(np.mean(pvet_pa_batch, axis=-1))  # (bs,)
            if return_transformed_points:
                return_dict['pred_reposed_vertices_pa'] = pred_reposed_vertices_pa

        if 'mpjpes' in self.metrics_to_track:
            mpjpe_batch = np.linalg.norm(pred_dict['joints3D'] - target_dict['joints3D'],
                                         axis=-1)  # (bsize, 14)
            self.metric_sums['mpjpes'] += np.sum(mpjpe_batch)  # scalar
            self.per_frame_metrics['mpjpes'].append(np.mean(mpjpe_batch, axis=-1))  # (bs,)

        # Scale and translation correction
        if 'mpjpes_sc' in self.metrics_to_track:
            pred_joints3D_h36mlsp = pred_dict['joints3D']  # (bsize, 14, 3)
            target_joints3D_h36mlsp = target_dict['joints3D']  # (bsize, 14, 3)
            pred_joints3D_h36mlsp_sc = scale_and_translation_transform_batch(pred_joints3D_h36mlsp,
                                                                             target_joints3D_h36mlsp)
            mpjpe_sc_batch = np.linalg.norm(pred_joints3D_h36mlsp_sc - target_joints3D_h36mlsp,
                                            axis=-1)  # (bsize, 14)
            self.metric_sums['mpjpes_sc'] += np.sum(mpjpe_sc_batch)  # scalar
            self.per_frame_metrics['mpjpes_sc'].append(np.mean(mpjpe_sc_batch, axis=-1))  # (bs,)
            if return_transformed_points:
                return_dict['pred_joints3D_h36mlsp_sc'] = pred_joints3D_h36mlsp_sc

        # Procrustes analysis
        if 'mpjpes_pa' in self.metrics_to_track:
            pred_joints3D_h36mlsp = pred_dict['joints3D']  # (bsize, 14, 3)
            target_joints3D_h36mlsp = target_dict['joints3D']  # (bsize, 14, 3)
            pred_joints3D_h36mlsp_pa = procrustes_analysis_batch(pred_joints3D_h36mlsp,
                                                                 target_joints3D_h36mlsp)
            mpjpe_pa_batch = np.linalg.norm(pred_joints3D_h36mlsp_pa - target_joints3D_h36mlsp,
                                            axis=-1)  # (bsize, 14)
            self.metric_sums['mpjpes_pa'] += np.sum(mpjpe_pa_batch)  # scalar
            self.per_frame_metrics['mpjpes_pa'].append(np.mean(mpjpe_pa_batch, axis=-1))  # (bs,)
            if return_transformed_points:
                return_dict['pred_joints3D_h36mlsp_pa'] = pred_joints3D_h36mlsp_pa

        if 'pose_mses' in self.metrics_to_track:
            self.metric_sums['pose_mses'] += np.sum((pred_dict['pose_params_rot_matrices'] -
                                                              target_dict['pose_params_rot_matrices']) ** 2)

        if 'shape_mses' in self.metrics_to_track:
            self.metric_sums['shape_mses'] += np.sum((pred_dict['shape_params'] -
                                                               target_dict['shape_params']) ** 2)

        if 'joints2D_l2es' in self.metrics_to_track:
            pred_joints2D_coco = pred_dict['joints2D']
            target_joints2D_coco = target_dict['joints2D']
            joints2D_l2e_batch = np.linalg.norm(pred_joints2D_coco - target_joints2D_coco,
                                                axis=-1)  # (bsize, num_joints)
            self.metric_sums['joints2D_l2es'] += np.sum(joints2D_l2e_batch)  # scalar
            self.per_frame_metrics['joints2D_l2es'].append(np.mean(joints2D_l2e_batch, axis=-1))  # (bs,)

        if 'silhouette_ious' in self.metrics_to_track:
            pred_silhouettes = pred_dict['silhouettes']
            target_silhouettes = target_dict['silhouettes']
            true_positive = np.logical_and(pred_silhouettes,
                                           target_silhouettes)
            false_positive = np.logical_and(pred_silhouettes,
                                            np.logical_not(target_silhouettes))
            true_negative = np.logical_and(np.logical_not(pred_silhouettes),
                                           np.logical_not(target_silhouettes))
            false_negative = np.logical_and(np.logical_not(pred_silhouettes),
                                            target_silhouettes)
            num_tp = np.sum(true_positive, axis=(1, 2))  # (bs,)
            num_fp = np.sum(false_positive, axis=(1, 2))
            num_tn = np.sum(true_negative, axis=(1, 2))
            num_fn = np.sum(false_negative, axis=(1, 2))
            self.metric_sums['num_true_positives'] += np.sum(num_tp)  # scalar
            self.metric_sums['num_false_positives'] += np.sum(num_fp)
            self.metric_sums['num_true_negatives'] += np.sum(num_tn)
            self.metric_sums['num_false_negatives'] += np.sum(num_fn)
            iou_per_frame = num_tp/(num_tp + num_fp + num_fn)
            self.per_frame_metrics['silhouette_ious'].append(iou_per_frame)  # (bs,)

        if return_transformed_points:
            return return_dict

    def compute_final_metrics(self):
        final_metrics = {}
        for metric_type in self.metrics_to_track:
            if metric_type == 'silhouette_ious':
                iou = self.metric_sums['num_true_positives'] / \
                      (self.metric_sums['num_true_positives'] +
                       self.metric_sums['num_false_negatives'] +
                       self.metric_sums['num_false_positives'])
                final_metrics['silhouette_ious'] = iou
            else:
                if 'pve' in metric_type:
                    num_per_sample = 6890
                elif 'mpjpe' in metric_type:
                    num_per_sample = 14
                elif 'joints2D' in metric_type:
                    num_per_sample = 17
                elif 'shape_mse' in metric_type:
                    num_per_sample = 10
                elif 'pose_mse' in metric_type:
                    num_per_sample = 24 * 3 * 3

                final_metrics[metric_type] = self.metric_sums[metric_type] / (self.total_samples * num_per_sample)
        print(final_metrics)

        if self.save_per_frame_metrics:
            for metric_type in self.metrics_to_track:
                per_frame = np.concatenate(self.per_frame_metrics[metric_type], axis=0)
                np.save(os.path.join(self.save_path, metric_type+'_per_frame.npy'), per_frame)