import numpy as np
import pickle

from utils.eval_utils import procrustes_analysis_batch, scale_and_translation_transform_batch
from utils.joints2d_utils import undo_keypoint_normalisation


class TrainingLossesAndMetricsTracker:
    """
    Tracks training and validation losses (both total and per-task) and metrics during
    training. Updates loss and metrics history at end of each epoch.
    """
    def __init__(self, losses_to_track, metrics_to_track, img_wh, log_path,
                 load_logs=False, current_epoch=None):

        self.all_per_task_loss_types = ['train_verts_losses', 'val_verts_losses',
                                        'train_shape_params_losses', 'val_shape_params_losses',
                                        'train_pose_params_losses', 'val_pose_params_losses',
                                        'train_joints2D_losses', 'val_joints2D_losses',
                                        'train_joints3D_losses', 'val_joints3D_losses']

        self.all_metrics_types = ['train_pves', 'val_pves',
                                  'train_pves_sc', 'val_pves_sc',
                                  'train_pves_pa', 'val_pves_pa',
                                  'train_pve-ts', 'val_pve-ts',
                                  'train_pve-ts_sc', 'val_pve-ts_sc',
                                  'train_pve-ts_pa', 'val_pve-ts_pa',
                                  'train_mpjpes', 'val_mpjpes',
                                  'train_mpjpes_sc', 'val_mpjpes_sc',
                                  'train_mpjpes_pa', 'val_mpjpes_pa',
                                  'train_pose_mses', 'val_pose_mses',
                                  'train_shape_mses', 'val_shape_mses',
                                  'train_joints2D_l2es', 'val_joints2D_l2es']

        self.losses_to_track = losses_to_track
        self.metrics_to_track = metrics_to_track
        self.img_wh = img_wh
        self.log_path = log_path

        if load_logs:
            self.history = self.load_history(log_path, current_epoch)
        else:
            self.history = {'train_losses': [], 'val_losses': []}
            for loss_type in self.all_per_task_loss_types:
                self.history[loss_type] = []
            for metric_type in self.all_metrics_types:
                self.history[metric_type] = []

        self.loss_metric_sums = None
        print('Metrics tracker initialised.')

    def load_history(self, load_log_path, current_epoch):
        """
        Loads loss (total and per-task) and metric history up to current epoch given the path
        to a log file. If per-task losses or metrics are missing from log file, fill with 0s.
        This is used if resuming a previous training run that was interrupted.
        """
        with open(load_log_path, 'rb') as f:
            history = pickle.load(f)

        history['train_losses'] = history['train_losses'][:current_epoch]
        history['val_losses'] = history['val_losses'][:current_epoch]

        # For each task, if per-task loss type in history, load up to current epoch.
        # Else, fill with 0s up to current epoch.
        for loss_type in self.all_per_task_loss_types:
            if loss_type in history.keys():
                history[loss_type] = history[loss_type][:current_epoch]
            else:
                history[loss_type] = [0.0] * current_epoch
                print(loss_type, 'filled with zeros up to epoch',
                      current_epoch)

        for metric_type in self.all_metrics_types:
            if metric_type in history.keys():
                history[metric_type] = history[metric_type][:current_epoch]
            else:
                history[metric_type] = [0.0] * current_epoch
                print(metric_type, 'filled with zeros up to epoch',
                      current_epoch)

        for key in history.keys():
            assert len(history[key]) == current_epoch, \
                "{} elements in {} list when current epoch is {}".format(
                    str(len(history[key])),
                    key,
                    str(current_epoch))
        print('Logs loaded from', load_log_path)

        return history

    def initialise_loss_metric_sums(self):
        self.loss_metric_sums = {'train_losses': 0., 'val_losses': 0.,
                                 'train_num_samples': 0, 'val_num_samples': 0}

        for loss_type in self.all_per_task_loss_types:
            self.loss_metric_sums[loss_type] = 0.

        for metric_type in self.all_metrics_types:
            self.loss_metric_sums[metric_type] = 0.

    def update_per_batch(self, split, loss, task_losses_dict, pred_dict, target_dict,
                         num_inputs_in_batch,
                         pred_reposed_vertices=None, target_reposed_vertices=None):
        assert split in ['train', 'val'], "Invalid split in metric tracker batch update."

        if any(['pve-ts' in metric_type for metric_type in self.metrics_to_track]):
            assert (pred_reposed_vertices is not None) and \
                   (target_reposed_vertices is not None), \
                "Need to pass reposed vertices to metric tracker batch update."
            pred_reposed_vertices = pred_reposed_vertices.cpu().detach().numpy()
            target_reposed_vertices = target_reposed_vertices.cpu().detach().numpy()

        for key in pred_dict.keys():
            pred_dict[key] = pred_dict[key].cpu().detach().numpy()
            target_dict[key] = target_dict[key].cpu().detach().numpy()

        # -------- Update loss sums --------
        self.loss_metric_sums[split + '_losses'] += loss.item() * num_inputs_in_batch
        self.loss_metric_sums[split + '_num_samples'] += num_inputs_in_batch

        for loss_on in self.losses_to_track:
            self.loss_metric_sums[split+'_'+loss_on+'_losses'] += task_losses_dict[loss_on].item() \
                                                                  * num_inputs_in_batch

        # -------- Update metrics sums --------
        if 'pves' in self.metrics_to_track:
            pve_batch = np.linalg.norm(pred_dict['verts'] - target_dict['verts'],
                                       axis=-1)  # (bsize, 6890)
            self.loss_metric_sums[split + '_pves'] += np.sum(pve_batch)  # scalar

        # Scale and translation correction
        if 'pves_sc' in self.metrics_to_track:
            pred_vertices = pred_dict['verts']  # (bsize, 6890, 3)
            target_vertices = target_dict['verts']  # (bsize, 6890, 3)
            pred_vertices_sc = scale_and_translation_transform_batch(pred_vertices,
                                                                     target_vertices)
            pve_sc_batch = np.linalg.norm(pred_vertices_sc - target_vertices,
                                          axis=-1)  # (bs, 6890)
            self.loss_metric_sums[split + '_pves_sc'] += np.sum(pve_sc_batch)  # scalar

        # Procrustes analysis
        if 'pves_pa' in self.metrics_to_track:
            pred_vertices = pred_dict['verts']  # (bsize, 6890, 3)
            target_vertices = target_dict['verts']  # (bsize, 6890, 3)
            pred_vertices_pa = procrustes_analysis_batch(pred_vertices, target_vertices)
            pve_pa_batch = np.linalg.norm(pred_vertices_pa - target_vertices,
                                          axis=-1)  # (bsize, 6890)
            self.loss_metric_sums[split + '_pves_pa'] += np.sum(pve_pa_batch)  # scalar

        # Reposed
        if 'pve-ts' in self.metrics_to_track:
            pvet_batch = np.linalg.norm(pred_reposed_vertices - target_reposed_vertices,
                                        axis=-1)
            self.loss_metric_sums[split + '_pve-ts'] += np.sum(pvet_batch)

        # Reposed + Scale and translation correction
        if 'pve-ts_sc' in self.metrics_to_track:
            pred_reposed_vertices_sc = scale_and_translation_transform_batch(pred_reposed_vertices,
                                                                             target_reposed_vertices)
            pvet_sc_batch = np.linalg.norm(pred_reposed_vertices_sc - target_reposed_vertices,
                                           axis=-1)  # (bs, 6890)
            self.loss_metric_sums[split + '_pve-ts_sc'] += np.sum(pvet_sc_batch)  # scalar

        # Reposed + Procrustes analysis - this doesn't make practical sense for reposed.
        if 'pve-ts_pa' in self.metrics_to_track:
            pred_reposed_vertices_pa = procrustes_analysis_batch(pred_reposed_vertices,
                                                                 target_reposed_vertices)
            pvet_pa_batch = np.linalg.norm(pred_reposed_vertices_pa - target_reposed_vertices,
                                           axis=-1)  # (bsize, 6890)
            self.loss_metric_sums[split + '_pve-ts_pa'] += np.sum(pvet_pa_batch)  # scalar

        if 'mpjpes' in self.metrics_to_track:
            mpjpe_batch = np.linalg.norm(pred_dict['joints3D'] - target_dict['joints3D'],
                                         axis=-1)  # (bsize, 14)
            self.loss_metric_sums[split + '_mpjpes'] += np.sum(mpjpe_batch)  # scalar

        # Scale and translation correction
        if 'mpjpes_sc' in self.metrics_to_track:
            pred_joints3D_h36mlsp = pred_dict['joints3D']  # (bsize, 14, 3)
            target_joints3D_h36mlsp = target_dict['joints3D']  # (bsize, 14, 3)
            pred_joints3D_h36mlsp_sc = scale_and_translation_transform_batch(pred_joints3D_h36mlsp,
                                                                             target_joints3D_h36mlsp)
            mpjpe_sc_batch = np.linalg.norm(pred_joints3D_h36mlsp_sc - target_joints3D_h36mlsp,
                                            axis=-1)  # (bsize, 14)
            self.loss_metric_sums[split + '_mpjpes_sc'] += np.sum(mpjpe_sc_batch)  # scalar

        # Procrustes analysis
        if 'mpjpes_pa' in self.metrics_to_track:
            pred_joints3D_h36mlsp = pred_dict['joints3D']  # (bsize, 14, 3)
            target_joints3D_h36mlsp = target_dict['joints3D']  # (bsize, 14, 3)
            pred_joints3D_h36mlsp_pa = procrustes_analysis_batch(pred_joints3D_h36mlsp,
                                                                 target_joints3D_h36mlsp)
            mpjpe_pa_batch = np.linalg.norm(pred_joints3D_h36mlsp_pa - target_joints3D_h36mlsp,
                                            axis=-1)  # (bsize, 14)
            self.loss_metric_sums[split + '_mpjpes_pa'] += np.sum(mpjpe_pa_batch)  # scalar

        if 'pose_mses' in self.metrics_to_track:
            self.loss_metric_sums[split + '_pose_mses'] += np.sum((pred_dict['pose_params_rot_matrices'] -
                                                                   target_dict['pose_params_rot_matrices']) ** 2)

        if 'shape_mses' in self.metrics_to_track:
            self.loss_metric_sums[split + '_shape_mses'] += np.sum((pred_dict['shape_params'] -
                                                                    target_dict['shape_params']) ** 2)

        if 'joints2D_l2es' in self.metrics_to_track:
            pred_joints2D_coco = pred_dict['joints2D']
            target_joints2D_coco = target_dict['joints2D']
            pred_joints2D_coco = undo_keypoint_normalisation(pred_joints2D_coco,
                                                             self.img_wh)
            joints2D_l2e_batch = np.linalg.norm(pred_joints2D_coco - target_joints2D_coco,
                                                axis=-1)  # (bsize, num_joints)
            self.loss_metric_sums[split + '_joints2D_l2es'] += np.sum(joints2D_l2e_batch)  # scalar

    def update_per_epoch(self):
        self.history['train_losses'].append(self.loss_metric_sums['train_losses'] /
                                            self.loss_metric_sums['train_num_samples'])
        self.history['val_losses'].append(self.loss_metric_sums['val_losses'] /
                                          self.loss_metric_sums['val_num_samples'])

        # For each task, if tracking per-task loss, append loss per sample for current epoch
        # to loss history. Else, append 0.
        for loss_type in self.all_per_task_loss_types:
            split = loss_type.split('_')[0]
            loss_on = loss_type[loss_type.find('_')+1:loss_type.find('_losses')]
            if loss_on in self.losses_to_track:
                self.history[loss_type].append(self.loss_metric_sums[loss_type] /
                                               self.loss_metric_sums[split + '_num_samples'])
            else:
                self.history[loss_type].append(0.)

        # For each metric, if tracking metric, append metric per sample for current epoch to
        # loss history. Else, append 0.
        for metric_type in self.all_metrics_types:
            split = metric_type.split('_')[0]

            if metric_type[metric_type.find('_')+1:] in self.metrics_to_track:
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

                self.history[metric_type].append(
                    self.loss_metric_sums[metric_type] /
                    (self.loss_metric_sums[split+'_num_samples'] * num_per_sample))

        # Print end of epoch losses and metrics.
        print('Finished epoch.')
        print('Train Loss: {:.5f}, Val Loss: {:.5f}'.format(self.history['train_losses'][-1],
                                                            self.history['val_losses'][-1]))
        for metric in self.metrics_to_track:
            print('Train {}: {:.5f}, Val {}: {:.5f}'.format(metric,
                                                            self.history['train_'+metric][-1],
                                                            metric,
                                                            self.history['val_'+metric][-1]))

        # Dump history to log file
        with open(self.log_path, 'wb') as f_out:
            pickle.dump(self.history, f_out)

    def determine_save_model_weights_this_epoch(self, save_val_metrics, best_epoch_val_metrics):
        save_model_weights_this_epoch = True
        for metric in save_val_metrics:
            if self.history['val_'+metric][-1] > best_epoch_val_metrics[metric]:
                save_model_weights_this_epoch = False
                break

        return save_model_weights_this_epoch

