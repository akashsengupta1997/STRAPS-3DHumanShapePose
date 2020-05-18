import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticTrainingDataset(Dataset):
    """
    Simple dataset that loads a npz file with 3 arrays containing:
    1) file names (from original datasets)
    2) SMPL pose parameters
    3) SMPL shape parameters.

    Returns dict with SMPL pose and shape (not augmented) as torch tensors.
    """
    def __init__(self,
                 npz_path,
                 params_from='all'):

        assert params_from in ['all', 'h36m', 'up3d', '3dpw', 'not_amass']

        data = np.load(npz_path)
        self.fnames = data['fnames']
        self.poses = data['poses']
        self.shapes = data['shapes']

        if params_from != 'all':
            if params_from == 'not_amass':
                indices = [i for i, x in enumerate(self.fnames)
                           if x.startswith('h36m') or x.startswith('up3d')
                           or x.startswith('3dpw')]
                self.fnames = [self.fnames[i] for i in indices]
                self.poses = [self.poses[i] for i in indices]
                self.shapes = [self.shapes[i] for i in indices]
            else:
                indices = [i for i, x in enumerate(self.fnames) if x.startswith(params_from)]
                self.fnames = [self.fnames[i] for i in indices]
                self.poses = [self.poses[i] for i in indices]
                self.shapes = [self.shapes[i] for i in indices]

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        pose = self.poses[index]
        shape = self.shapes[index]
        assert pose.shape == (72,) and shape.shape == (10,), \
            "Poses and shapes are wrong: {}, {}, {}".format(self.fnames[index],
                                                            pose.shape, shape.shape)

        pose = torch.from_numpy(pose.astype(np.float32))
        shape = torch.from_numpy(shape.astype(np.float32))

        return {'pose': pose,
                'shape': shape}
