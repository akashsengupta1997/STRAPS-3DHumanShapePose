import torch
import torch.nn as nn
import numpy as np

import config


class IEFModule(nn.Module):
    """
    Iterative error feedback module that regresses SMPL body model parameters (and
    weak-perspective camera parameters) given input features.
    """
    def __init__(self, fc_layers_neurons, in_features, num_output_params, iterations=3):
        super(IEFModule, self).__init__()

        self.fc1 = nn.Linear(in_features + num_output_params, fc_layers_neurons[0])
        self.fc2 = nn.Linear(fc_layers_neurons[0], fc_layers_neurons[1])
        self.fc3 = nn.Linear(fc_layers_neurons[1], num_output_params)
        self.relu = nn.ReLU(inplace=True)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.zeros_(self.fc3.bias)

        self.ief_layers = nn.Sequential(self.fc1,
                                        self.relu,
                                        self.fc2,
                                        self.relu,
                                        self.fc3)

        self.iterations = iterations

        self.num_output_params = num_output_params
        self.initial_params_estimate = self.load_mean_params_6d_pose(config.SMPL_MEAN_PARAMS_PATH)

    def load_mean_params_6d_pose(self, mean_params_path):
        mean_smpl = np.load(mean_params_path)
        mean_pose = mean_smpl['pose']
        mean_shape = mean_smpl['shape']

        mean_params = np.zeros(3 + 24*6 + 10)
        mean_params[3:] = np.concatenate((mean_pose, mean_shape))

        # Set initial weak-perspective camera parameters - [s, tx, ty]
        mean_params[0] = 0.9  # Initialise scale to 0.9
        mean_params[1] = 0.0
        mean_params[2] = 0.0

        return torch.from_numpy(mean_params.astype(np.float32)).float()

    def forward(self, img_features):
        batch_size = img_features.size(0)

        params_estimate = self.initial_params_estimate.repeat([batch_size, 1])
        params_estimate = params_estimate.to(img_features.device)

        state = torch.cat([img_features, params_estimate], dim=1)
        for i in range(self.iterations):
            delta = self.ief_layers(state)
            delta = delta * self.scaledown
            params_estimate += delta
            state = torch.cat([img_features, params_estimate], dim=1)

        if self.num_output_params == (3 + 24*3 + 10):
            cam_params = params_estimate[:, :3]
            pose_params = params_estimate[:, 3:3+24*3]
            shape_params = params_estimate[:, 3+24*3:]
        elif self.num_output_params == (3 + 24*6 + 10):
            cam_params = params_estimate[:, :3]
            pose_params = params_estimate[:, 3:3 + 24*6]
            shape_params = params_estimate[:, 3 + 24*6:]

        return cam_params, pose_params, shape_params




