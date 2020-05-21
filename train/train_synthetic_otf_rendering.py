import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from smplx.lbs import batch_rodrigues
from tqdm import tqdm

from metrics.train_loss_and_metrics_tracker import TrainingLossesAndMetricsTracker

from utils.checkpoint_utils import load_training_info_from_checkpoint

from utils.cam_utils import perspective_project_torch, orthographic_project_torch
from utils.rigid_transform_utils import rot6d_to_rotmat
from utils.label_conversions import convert_multiclass_to_binary_labels_torch, \
    convert_2Djoints_to_gaussian_heatmaps_torch
from utils.joints2d_utils import check_joints2d_visibility_torch
from utils.image_utils import batch_crop_seg_to_bounding_box, batch_resize

from augmentation.smpl_augmentation import augment_smpl
from augmentation.cam_augmentation import augment_cam_t
from augmentation.proxy_rep_augmentation import augment_proxy_representation, \
    random_verts2D_deviation

import config


def train_synthetic_otf_rendering(device,
                                  regressor,
                                  smpl_model,
                                  nmr_parts_renderer,
                                  train_dataset,
                                  val_dataset,
                                  criterion,
                                  optimiser,
                                  batch_size,
                                  num_epochs,
                                  img_wh,
                                  smpl_augment_params,
                                  cam_augment_params,
                                  bbox_augment_params,
                                  proxy_rep_augment_params,
                                  mean_cam_t,
                                  cam_K,
                                  cam_R,
                                  model_save_path,
                                  log_path,
                                  losses_to_track,
                                  metrics_to_track,
                                  save_val_metrics,
                                  epochs_per_save=10,
                                  checkpoint=None,
                                  num_workers=0,
                                  pin_memory=False):

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=True, num_workers=num_workers,
                                  pin_memory=pin_memory)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                drop_last=True, num_workers=num_workers,
                                pin_memory=pin_memory)

    # Ensure that all metrics used as model save conditions are being tracked (i.e. that
    # save_val_metrics is a subset of metrics_to_track).
    temp = save_val_metrics.copy()
    if 'loss' in save_val_metrics:
        temp.remove('loss')
    assert set(temp).issubset(set(metrics_to_track)), \
        "Not all save-condition metrics are being tracked!"

    if checkpoint is not None:
        # Resuming training - note that current model and optimiser state dicts are loaded out
        # of train function (should be in run file).
        current_epoch, best_epoch, best_model_wts, best_epoch_val_metrics = \
            load_training_info_from_checkpoint(checkpoint, save_val_metrics)
        load_logs = True

    else:
        current_epoch = 0
        best_epoch_val_metrics = {}
        # metrics that decide whether to save model after each epoch or not
        for metric in save_val_metrics:
            best_epoch_val_metrics[metric] = np.inf
        best_epoch = current_epoch
        best_model_wts = copy.deepcopy(regressor.state_dict())
        load_logs = False

    # Instantiate metrics tracker.
    metrics_tracker = TrainingLossesAndMetricsTracker(losses_to_track=losses_to_track,
                                                      metrics_to_track=metrics_to_track,
                                                      img_wh=img_wh,
                                                      log_path=log_path,
                                                      load_logs=load_logs,
                                                      current_epoch=current_epoch)

    # Loading mean shape (for augmentation function)
    mean_smpl = np.load(config.SMPL_MEAN_PARAMS_PATH)
    mean_shape = torch.from_numpy(mean_smpl['shape']).float().to(device)

    # Starting training loop
    for epoch in range(current_epoch, num_epochs):
        print('\nEpoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        metrics_tracker.initialise_loss_metric_sums()

        # ################################################################################
        # ----------------------------------- TRAINING -----------------------------------
        # ################################################################################
        print('Training.')
        regressor.train()
        for batch_num, samples_batch in enumerate(tqdm(train_dataloader)):
            if batch_num == 100:
                break
            # ---------------- SYNTHETIC DATA GENERATION ----------------
            with torch.no_grad():
                # TARGET SMPL PARAMETERS
                target_pose = samples_batch['pose']
                target_shape = samples_batch['shape']
                target_pose = target_pose.to(device)
                target_shape = target_shape.to(device)
                num_train_inputs_in_batch = target_pose.shape[0]  # Same as bs since drop_last=True

                # SMPL AND CAM AUGMENTATION
                target_shape, target_pose_rotmats, target_glob_rotmats = augment_smpl(
                    target_shape,
                    target_pose[:, 3:],
                    target_pose[:, :3],
                    mean_shape,
                    smpl_augment_params)
                target_cam_t = augment_cam_t(mean_cam_t,
                                             xy_std=cam_augment_params['xy_std'],
                                             delta_z_range=cam_augment_params['delta_z_range'])

                # TARGET VERTICES AND JOINTS
                target_smpl_output = smpl_model(body_pose=target_pose_rotmats,
                                                global_orient=target_glob_rotmats,
                                                betas=target_shape,
                                                pose2rot=False)
                target_vertices = target_smpl_output.vertices
                target_joints_all = target_smpl_output.joints
                target_joints_h36m = target_joints_all[:, config.ALL_JOINTS_TO_H36M_MAP, :]
                target_joints_h36mlsp = target_joints_h36m[:, config.H36M_TO_J14, :]
                target_joints_coco = target_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]
                target_joints2d_coco = perspective_project_torch(target_joints_coco, cam_R,
                                                                 target_cam_t,
                                                                 cam_K=cam_K)
                target_reposed_smpl_output = smpl_model(betas=target_shape)
                target_reposed_vertices = target_reposed_smpl_output.vertices

                if proxy_rep_augment_params['deviate_verts2D']:
                    # Vertex noise augmentation to give noisy proxy representation edges
                    target_vertices_for_rendering = random_verts2D_deviation(target_vertices,
                                                                             delta_verts2d_dev_range=proxy_rep_augment_params['delta_verts2d_dev_range'])
                else:
                    target_vertices_for_rendering = target_vertices

                # INPUT PROXY REPRESENTATION GENERATION
                input = nmr_parts_renderer(target_vertices_for_rendering, target_cam_t)

                # BBOX AUGMENTATION AND CROPPING
                if bbox_augment_params['crop_input']:
                    # Crop inputs according to bounding box
                    # + add random scale and centre augmentation
                    input = input.cpu().detach().numpy()
                    target_joints2d_coco = target_joints2d_coco.cpu().detach().numpy()
                    all_cropped_segs, all_cropped_joints2D = batch_crop_seg_to_bounding_box(
                        input, target_joints2d_coco,
                        orig_scale_factor=bbox_augment_params['mean_scale_factor'],
                        delta_scale_range=bbox_augment_params['delta_scale_range'],
                        delta_centre_range=bbox_augment_params['delta_centre_range'])
                    resized_input, resized_joints2D = batch_resize(all_cropped_segs, all_cropped_joints2D, img_wh)
                    input = torch.from_numpy(resized_input).float().to(device)
                    target_joints2d_coco = torch.from_numpy(resized_joints2D).float().to(device)

                # PROXY REPRESENTATION AUGMENTATION
                input, target_joints2d_coco_input = augment_proxy_representation(input,
                                                                                 target_joints2d_coco,
                                                                                 proxy_rep_augment_params)

                # FINAL INPUT PROXY REPRESENTATION GENERATION WITH JOINT HEATMAPS
                input = convert_multiclass_to_binary_labels_torch(input)
                input = input.unsqueeze(1)
                j2d_heatmaps = convert_2Djoints_to_gaussian_heatmaps_torch(target_joints2d_coco_input,
                                                                           img_wh)
                input = torch.cat([input, j2d_heatmaps], dim=1)

            # ---------------- FORWARD PASS ----------------
            # (gradients being computed from here on)
            pred_cam_wp, pred_pose, pred_shape = regressor(input)

            # Convert pred pose to rotation matrices
            if pred_pose.shape[-1] == 24*3:
                pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
            elif pred_pose.shape[-1] == 24*6:
                pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

            # PREDICTED VERTICES AND JOINTS
            pred_smpl_output = smpl_model(body_pose=pred_pose_rotmats[:, 1:],
                                          global_orient=pred_pose_rotmats[:, 0].unsqueeze(1),
                                          betas=pred_shape,
                                          pose2rot=False)
            pred_vertices = pred_smpl_output.vertices
            pred_joints_all = pred_smpl_output.joints
            pred_joints_h36m = pred_joints_all[:, config.ALL_JOINTS_TO_H36M_MAP, :]
            pred_joints_h36mlsp = pred_joints_h36m[:, config.H36M_TO_J14, :]
            pred_joints_coco = pred_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]
            pred_joints2d_coco = orthographic_project_torch(pred_joints_coco, pred_cam_wp)
            pred_reposed_smpl_output = smpl_model(betas=pred_shape)
            pred_reposed_vertices = pred_reposed_smpl_output.vertices

            # ---------------- LOSS ----------------
            # Concatenate target pose and global rotmats for loss function
            target_pose_rotmats = torch.cat([target_glob_rotmats, target_pose_rotmats],
                                            dim=1)
            # Check joints visibility
            target_joints2d_vis_coco = check_joints2d_visibility_torch(target_joints2d_coco,
                                                                       img_wh)

            pred_dict_for_loss = {'joints2D': pred_joints2d_coco,
                                  'verts': pred_vertices,
                                  'shape_params': pred_shape,
                                  'pose_params_rot_matrices': pred_pose_rotmats,
                                  'joints3D': pred_joints_h36mlsp}
            target_dict_for_loss = {'joints2D': target_joints2d_coco,
                                    'verts': target_vertices,
                                    'shape_params': target_shape,
                                    'pose_params_rot_matrices': target_pose_rotmats,
                                    'joints3D': target_joints_h36mlsp,
                                    'vis': target_joints2d_vis_coco}

            # ---------------- BACKWARD PASS ----------------
            optimiser.zero_grad()
            loss, task_losses_dict = criterion(target_dict_for_loss, pred_dict_for_loss)
            loss.backward()
            optimiser.step()

            # ---------------- TRACK LOSS AND METRICS ----------------
            metrics_tracker.update_per_batch('train', loss, task_losses_dict,
                                             pred_dict_for_loss, target_dict_for_loss,
                                             num_train_inputs_in_batch,
                                             pred_reposed_vertices=pred_reposed_vertices,
                                             target_reposed_vertices=target_reposed_vertices)

        # ##################################################################################
        # ----------------------------------- VALIDATION -----------------------------------
        # ##################################################################################
        print('Validation.')
        regressor.eval()
        with torch.no_grad():
            for batch_num, samples_batch in enumerate(tqdm(val_dataloader)):
                if batch_num == 100:
                    break
                # ---------------- SYNTHETIC DATA GENERATION ----------------
                # TARGET SMPL PARAMETERS
                target_pose = samples_batch['pose']
                target_shape = samples_batch['shape']
                target_pose = target_pose.to(device)
                target_shape = target_shape.to(device)
                num_val_inputs_in_batch = target_pose.shape[0]  # Same as bs since drop_last=True

                # TARGET VERTICES AND JOINTS
                target_smpl_output = smpl_model(body_pose=target_pose[:, 3:],
                                                global_orient=target_pose[:, :3],
                                                betas=target_shape)
                target_vertices = target_smpl_output.vertices
                target_joints_all = target_smpl_output.joints
                target_joints_h36m = target_joints_all[:, config.ALL_JOINTS_TO_H36M_MAP, :]
                target_joints_h36mlsp = target_joints_h36m[:, config.H36M_TO_J14, :]
                target_joints_coco = target_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]
                target_joints2d_coco = perspective_project_torch(target_joints_coco, cam_R,
                                                                 mean_cam_t,
                                                                 cam_K=cam_K)
                target_reposed_smpl_output = smpl_model(betas=target_shape)
                target_reposed_vertices = target_reposed_smpl_output.vertices

                # INPUT PROXY REPRESENTATION GENERATION
                input = nmr_parts_renderer(target_vertices, mean_cam_t)

                # BBOX AUGMENTATION AND CROPPING
                if bbox_augment_params['crop_input']:  # Crop inputs according to bounding box
                    input = input.cpu().detach().numpy()
                    target_joints2d_coco = target_joints2d_coco.cpu().detach().numpy()
                    all_cropped_segs, all_cropped_joints2D = batch_crop_seg_to_bounding_box(
                        input, target_joints2d_coco,
                        orig_scale_factor=bbox_augment_params['mean_scale_factor'],
                        delta_scale_range=None,
                        delta_centre_range=None)
                    resized_input, resized_joints2D = batch_resize(all_cropped_segs,
                                                                   all_cropped_joints2D,
                                                                   img_wh)
                    input = torch.from_numpy(resized_input).float().to(device)
                    target_joints2d_coco = torch.from_numpy(resized_joints2D).float().to(device)

                # FINAL INPUT PROXY REPRESENTATION GENERATION WITH JOINT HEATMAPS
                input = convert_multiclass_to_binary_labels_torch(input)
                input = input.unsqueeze(1)
                j2d_heatmaps = convert_2Djoints_to_gaussian_heatmaps_torch(target_joints2d_coco,
                                                                           img_wh)
                input = torch.cat([input, j2d_heatmaps], dim=1)

                # ---------------- FORWARD PASS ----------------
                pred_cam_wp, pred_pose, pred_shape = regressor(input)
                # Convert pred pose to rotation matrices
                if pred_pose.shape[-1] == 24 * 3:
                    pred_pose_rotmats = batch_rodrigues(pred_pose.contiguous().view(-1, 3))
                    pred_pose_rotmats = pred_pose_rotmats.view(-1, 24, 3, 3)
                elif pred_pose.shape[-1] == 24 * 6:
                    pred_pose_rotmats = rot6d_to_rotmat(pred_pose.contiguous()).view(-1, 24, 3, 3)

                # PREDICTED VERTICES AND JOINTS
                pred_smpl_output = smpl_model(body_pose=pred_pose_rotmats[:, 1:],
                                              global_orient=pred_pose_rotmats[:, 0].unsqueeze(1),
                                              betas=pred_shape,
                                              pose2rot=False)
                pred_vertices = pred_smpl_output.vertices
                pred_joints_all = pred_smpl_output.joints
                pred_joints_h36m = pred_joints_all[:, config.ALL_JOINTS_TO_H36M_MAP, :]
                pred_joints_h36mlsp = pred_joints_h36m[:, config.H36M_TO_J14, :]
                pred_joints_coco = pred_joints_all[:, config.ALL_JOINTS_TO_COCO_MAP, :]
                pred_joints2d_coco = orthographic_project_torch(pred_joints_coco, pred_cam_wp)
                pred_reposed_smpl_output = smpl_model(betas=pred_shape)
                pred_reposed_vertices = pred_reposed_smpl_output.vertices

                # ---------------- LOSS ----------------
                # Convert pose parameters to rotation matrices for loss function
                target_pose_rotmats = batch_rodrigues(target_pose.contiguous().view(-1, 3))
                target_pose_rotmats = target_pose_rotmats.view(-1, 24, 3, 3)

                # Check joints visibility
                target_joints2d_vis_coco = check_joints2d_visibility_torch(target_joints2d_coco, img_wh)

                pred_dict_for_loss = {'joints2D': pred_joints2d_coco,
                                      'verts': pred_vertices,
                                      'shape_params': pred_shape,
                                      'pose_params_rot_matrices': pred_pose_rotmats,
                                      'joints3D': pred_joints_h36mlsp}
                target_dict_for_loss = {'joints2D': target_joints2d_coco,
                                        'verts': target_vertices,
                                        'shape_params': target_shape,
                                        'pose_params_rot_matrices': target_pose_rotmats,
                                        'joints3D': target_joints_h36mlsp,
                                        'vis': target_joints2d_vis_coco}

                val_loss, val_task_losses_dict = criterion(target_dict_for_loss,
                                                           pred_dict_for_loss)

                # ---------------- TRACK LOSS AND METRICS ----------------
                metrics_tracker.update_per_batch('val', val_loss, val_task_losses_dict,
                                                 pred_dict_for_loss, target_dict_for_loss,
                                                 num_val_inputs_in_batch,
                                                 pred_reposed_vertices=pred_reposed_vertices,
                                                 target_reposed_vertices=target_reposed_vertices)

        # ----------------------- UPDATING LOSS AND METRICS HISTORY -----------------------
        metrics_tracker.update_per_epoch()

        # ----------------------------------- SAVING -----------------------------------
        save_model_weights_this_epoch = metrics_tracker.determine_save_model_weights_this_epoch(save_val_metrics,
                                                                                                best_epoch_val_metrics)

        if save_model_weights_this_epoch:
            for metric in save_val_metrics:
                best_epoch_val_metrics[metric] = metrics_tracker.history['val_' + metric][-1]
            print("Best epoch val metrics updated!")
            best_model_wts = copy.deepcopy(regressor.state_dict())
            best_epoch = epoch
            print("Best model weights updated!")

        if epoch % epochs_per_save == 0:
            # Saving current epoch num, best epoch num, best validation metrics (occurred in best
            # epoch num), current regressor state_dict, best regressor state_dict, current
            # optimiser state dict and current criterion state_dict (i.e. multi-task loss weights).
            save_dict = {'epoch': epoch,
                         'best_epoch': best_epoch,
                         'best_epoch_val_metrics': best_epoch_val_metrics,
                         'model_state_dict': regressor.state_dict(),
                         'best_model_state_dict': best_model_wts,
                         'optimiser_state_dict': optimiser.state_dict(),
                         'criterion_state_dict': criterion.state_dict()}
            torch.save(save_dict,
                       model_save_path + '_epoch{}'.format(epoch) + '.tar')
            print('Model saved! Best Val Metrics:\n',
                  best_epoch_val_metrics,
                  '\nin epoch {}'.format(best_epoch))

    print('Training Completed. Best Val Metrics:\n',
          best_epoch_val_metrics)

    regressor.load_state_dict(best_model_wts)
    return regressor
