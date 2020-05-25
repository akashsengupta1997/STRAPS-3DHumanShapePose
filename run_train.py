import os
import torch
import numpy as np
import torch.optim as optim

import config

from data.synthetic_training_dataset import SyntheticTrainingDataset

from models.regressor import SingleInputRegressor
from models.smpl_official import SMPL
from renderers.nmr_renderer import NMRRenderer

from losses.multi_task_loss import HomoscedasticUncertaintyWeightedMultiTaskLoss

from utils.model_utils import count_parameters
from utils.cam_utils import get_intrinsics_matrix

from train.train_synthetic_otf_rendering import train_synthetic_otf_rendering


# ----------------------- Device -----------------------
gpu = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('\nDevice: {}'.format(device))
print('GPU:', gpu)

# ----------------------- Dataloading settings -----------------------
num_workers = 4
pin_memory = True

# ----------------------- Network settings -----------------------
resnet_in_channels = 1 + 17  # single-channel silhouette + 17 joint heatmaps
resnet_layers = 18
ief_iters = 3
print("\nResNet in channels:", resnet_in_channels)
print("ResNet layers:", resnet_layers)
print("IEF Num iters:", ief_iters)

# ----------------------- Hyperparameters -----------------------
num_epochs = 1000
batch_size = 140
lr = 0.0001
epochs_per_save = 10
print("\nBatch size:", batch_size)
print("LR:", lr)
print("Image width/height:", config.REGRESSOR_IMG_WH)

# ----------------------- Loss settings -----------------------
losses_on = ['verts', 'shape_params', 'pose_params', 'joints2D', 'joints3D']
init_loss_weights = {'verts': 1.0, 'joints2D': 0.1, 'pose_params': 0.1, 'shape_params': 0.1,
                     'joints3D': 1.0}  # Initial loss weights - these will be updated during training.
losses_to_track = losses_on
normalise_joints_before_loss = True
silhouette_loss = ('silhouette' in losses_on)

print("\nLosses on:", losses_on)
print("Loss weights:", init_loss_weights)

# ----------------------- Metrics settings -----------------------
metrics_to_track = ['pves', 'pves_sc', 'pves_pa', 'pve-ts', 'pve-ts_sc', 'mpjpes', 'mpjpes_sc',
                    'mpjpes_pa', 'shape_mses', 'pose_mses', 'joints2D_l2es']
save_val_metrics = ['pves_pa', 'mpjpes_pa']
# ^ Main metrics which asses model validation performance - used to determine when to save model weights.
print("\nMetrics:", metrics_to_track)
print("Save val metrics:", save_val_metrics)

# ----------------------- Paths -----------------------
# Path to npz with training data.
train_path = 'data/amass_h36m_up3d_3dpw_train.npz'  # TODO change to file without h36m
# Path to npz with validation data.
val_path = 'data/h36m_up3d_3dpw_val.npz'  # TODO change to file without h36m

# Path to save model weights to (without .tar extension).
model_save_path = os.path.join('./checkpoints/model_checkpoint')
log_path = os.path.join('./logs/model_logs.pkl')

resume_from_epoch = None  # Epoch number from which to resume training (if applicable).
if resume_from_epoch is not None:
    checkpoint_path = model_save_path + '_epoch{}'.format(str(resume_from_epoch)) + '.tar'
else:
    checkpoint_path = None

print("\nTrain path:", train_path)
print("Val path:", val_path)
print("Model save path:", model_save_path)
print("Log save path:", log_path)

# ----------------------- Dataset -----------------------
train_dataset = SyntheticTrainingDataset(npz_path=train_path, params_from='not_amass')
val_dataset = SyntheticTrainingDataset(npz_path=val_path, params_from='all')
train_val_monitor_datasets = [train_dataset, val_dataset]
print("Training examples found:", len(train_dataset))
print("Validation examples found:", len(val_dataset))

# ----------------------- Models -----------------------
# Regressor
regressor = SingleInputRegressor(resnet_in_channels, resnet_layers, ief_iters=ief_iters)
num_params = count_parameters(regressor)
print("\nRegressor model Loaded. ", num_params, "trainable parameters.")

# SMPL model
smpl_model = SMPL(config.SMPL_MODEL_DIR,
                  batch_size=batch_size)

# Camera and NMR part/silhouette renderer
# Assuming camera rotation is identity (since it is dealt with by global_orients in SMPL)
mean_cam_t = np.array([0., 0.2, 42.])
mean_cam_t = torch.from_numpy(mean_cam_t).float().to(device)
mean_cam_t = mean_cam_t[None, :].expand(batch_size, -1)
cam_K = get_intrinsics_matrix(config.REGRESSOR_IMG_WH, config.REGRESSOR_IMG_WH, config.FOCAL_LENGTH)
cam_K = torch.from_numpy(cam_K.astype(np.float32)).to(device)
cam_K = cam_K[None, :, :].expand(batch_size, -1, -1)
cam_R = torch.eye(3).to(device)
cam_R = cam_R[None, :, :].expand(batch_size, -1, -1)
nmr_parts_renderer = NMRRenderer(batch_size,
                                 cam_K,
                                 cam_R,
                                 config.REGRESSOR_IMG_WH,
                                 rend_parts_seg=True)

regressor.to(device)
smpl_model.to(device)
nmr_parts_renderer.to(device)

# ----------------------- Augmentation -----------------------
# SMPL
augment_shape = True
delta_betas_distribution = 'normal'
delta_betas_std_vector = torch.tensor([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
                                      device=device).float()  # used if delta_betas_distribution is 'normal'
delta_betas_range = [-3., 3.]  # used if delta_betas_distribution is 'uniform'
smpl_augment_params = {'augment_shape': augment_shape,
                       'delta_betas_distribution': delta_betas_distribution,
                       'delta_betas_std_vector': delta_betas_std_vector,
                       'delta_betas_range': delta_betas_range}
# Camera
xy_std = 0.05
delta_z_range = [-5, 5]
cam_augment_params = {'xy_std': xy_std,
                      'delta_z_range': delta_z_range}
# BBox
crop_input = True
mean_scale_factor = 1.2
delta_scale_range = [-0.2, 0.2]
delta_centre_range = [-5, 5]
bbox_augment_params = {'crop_input': crop_input,
                       'mean_scale_factor': mean_scale_factor,
                       'delta_scale_range': delta_scale_range,
                       'delta_centre_range': delta_centre_range}
# Proxy Representation
remove_appendages = True
deviate_joints2D = True
deviate_verts2D = True
occlude_seg = True

remove_appendages_classes = [1, 2, 3, 4, 5, 6]
remove_appendages_probabilities = [0.1, 0.1, 0.1, 0.1, 0.05, 0.05]
delta_j2d_dev_range = [-8, 8]
delta_j2d_hip_dev_range = [-8, 8]
delta_verts2d_dev_range = [-0.01, 0.01]
occlude_probability = 0.5
occlude_box_dim = 48

proxy_rep_augment_params = {'remove_appendages': remove_appendages,
                            'deviate_joints2D': deviate_joints2D,
                            'deviate_verts2D': deviate_verts2D,
                            'occlude_seg': occlude_seg,
                            'remove_appendages_classes': remove_appendages_classes,
                            'remove_appendages_probabilities': remove_appendages_probabilities,
                            'delta_j2d_dev_range': delta_j2d_dev_range,
                            'delta_j2d_hip_dev_range': delta_j2d_hip_dev_range,
                            'delta_verts2d_dev_range': delta_verts2d_dev_range,
                            'occlude_probability': occlude_probability,
                            'occlude_box_dim': occlude_box_dim}

print('\nSMPL augment params:')
print(smpl_augment_params)
print('Cam augment params:')
print(cam_augment_params)
print('Crop input:', crop_input)
print('BBox augment params')
print(bbox_augment_params)
print('Proxy rep augment params')
print(proxy_rep_augment_params)

# ----------------------- Loss -----------------------
criterion = HomoscedasticUncertaintyWeightedMultiTaskLoss(losses_on,
                                                          init_loss_weights=init_loss_weights,
                                                          reduction='mean')
criterion.to(device)

# ----------------------- Optimiser -----------------------
params = list(regressor.parameters()) + list(criterion.parameters())
optimiser = optim.Adam(params, lr=lr)

# ----------------------- Resuming -----------------------
if checkpoint_path is not None:
    print('Resuming from:', checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    regressor.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
    criterion.load_state_dict(checkpoint['criterion_state_dict'])
else:
    checkpoint = None

train_synthetic_otf_rendering(device=device,
                              regressor=regressor,
                              smpl_model=smpl_model,
                              nmr_parts_renderer=nmr_parts_renderer,
                              train_dataset=train_dataset,
                              val_dataset=val_dataset,
                              criterion=criterion,
                              optimiser=optimiser,
                              batch_size=batch_size,
                              num_epochs=num_epochs,
                              smpl_augment_params=smpl_augment_params,
                              cam_augment_params=cam_augment_params,
                              bbox_augment_params=bbox_augment_params,
                              proxy_rep_augment_params=proxy_rep_augment_params,
                              mean_cam_t=mean_cam_t,
                              cam_K=cam_K,
                              cam_R=cam_R,
                              model_save_path=model_save_path,
                              log_path=log_path,
                              losses_to_track=losses_to_track,
                              metrics_to_track=metrics_to_track,
                              save_val_metrics=save_val_metrics,
                              epochs_per_save=epochs_per_save,
                              checkpoint=checkpoint,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
