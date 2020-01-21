import os

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from dq3d import quat, dualquat
import transforms3d as tf3d

from ArticulationModelLearning.magic.lstm.utils import quat_as_wxyz


class ArticulationDataset(Dataset):
    def __init__(self,
                 ntrain,
                 root_dir,
                 n_dof):
        super(ArticulationDataset, self).__init__()

        self.root_dir = root_dir
        self.labels_data = None
        self.length = ntrain
        self.n_dof = n_dof

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.labels_data is None:
            self.labels_data = h5py.File(os.path.join(self.root_dir, 'complete_data.hdf5'), 'r')

        # One Sample for us corresponds to one instantiation of an object type
        obj_data = self.labels_data['obj_' + str(idx).zfill(6)]

        # Load depth image
        depth_imgs = torch.tensor(obj_data['depth_imgs'])
        depth_imgs.unsqueeze_(1).float()
        depth_imgs = torch.cat((depth_imgs, depth_imgs, depth_imgs), dim=1)

        # # Load labels
        ref_frame = obj_data['joint_frame_in_world']
        q_vals = obj_data['q']

        l_hat = tf3d.quaternions.rotate_vector(np.array([0., 0., 1.]), quat_as_wxyz(ref_frame[3:]))
        m = np.cross(ref_frame[:3], l_hat)

        """Currently  considering only microwave"""
        d = np.array([0.])

        # Generating labels in screw notation: label := <l_hat, m, theta, d> = <3, 3, 1, 1>
        label = np.empty((len(q_vals), 8))

        # for i, q in enumerate(q_vals):
        #     label[i, :] = np.concatenate((l_hat, m, q, d))

        # Construct dual quaternions
        dq = dualquat.zeros()
        for i, q in enumerate(q_vals):
            dualquat.from_screw(dq, q[0], d, l_hat, m)
            label[i, :] = np.concatenate((dq.real.data, dq.dual.data))

        # ref_frame = obj_data['reference_frame_in_world']
        # moving_frames = obj_data['moving_frame_in_ref_frame']
        # dq = dualquat(quat(ref_frame[3:]), ref_frame[:3])
        # label = np.concatenate((dq.real.data, dq.dual.data))    # quat: x,y,z,w
        #
        # for pt in moving_frames:
        #     dq = dualquat(quat(pt[3:]), pt[:3])
        #     label = np.vstack((label, np.concatenate((dq.real.data, dq.dual.data))))

        label = torch.from_numpy(label).float()
        sample = {'depth': depth_imgs,
                  'label': label}

        return sample
