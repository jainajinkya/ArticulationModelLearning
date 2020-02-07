import os
from itertools import permutations, combinations

import dq3d
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from dq3d import quat, dualquat
import transforms3d as tf3d

from ArticulationModelLearning.magic.lstm.utils import quat_as_wxyz, transform_to_screw, quat_as_xyzw
from SyntheticArticulatedData.generation.utils import change_frames


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
        moving_body_poses = obj_data['moving_frame_in_world']

        label = np.empty((len(moving_body_poses) - 1, 8))

        for i in range(len(moving_body_poses) - 1):
            pt1 = moving_body_poses[i, :]
            pt2 = moving_body_poses[i + 1, :]
            pt1_T_pt2 = change_frames(pt1, pt2)

            # Generating labels in screw notation: label := <l_hat, m, theta, d> = <3, 3, 1, 1>
            l_hat, m, theta, d = transform_to_screw(translation=pt1_T_pt2[:3],
                                                    quat_in_wxyz=pt1_T_pt2[3:])
            # print("Screw notation: {}\t{}\t{}\t{}".format(np.round(l_hat, 4),
            #                                               np.round(m, 4),
            #                                               np.round(theta, 4),
            #                                               np.round(d, 4)))
            label[i, :] = np.concatenate((l_hat, m, [theta], [d]))

            # # Generating labels in dual quaternions
            # dq = dq3d.dualquat(dq3d.quat(quat_as_xyzw(pt1_T_pt2[3:])), pt1_T_pt2[:3])
            # # print("Dual Quaternion: {}\t{}".format(np.round(dq.real.data, 4),
            # #                                        np.round(dq.dual.data, 4)))
            # label[i, :] = np.concatenate((np.array([dq.real.w, dq.real.x, dq.real.y, dq.real.z]),
            #                               np.array([dq.dual.w, dq.dual.x, dq.dual.y, dq.dual.z])))

        label = torch.from_numpy(label).float()
        sample = {'depth': depth_imgs,
                  'label': label}

        return sample


''' LSTM + 2 images'''


class ArticulationDatasetV1(Dataset):
    def __init__(self,
                 ntrain,
                 root_dir,
                 n_dof):
        super(ArticulationDatasetV1, self).__init__()

        self.root_dir = root_dir
        self.labels_data = None
        self.length = ntrain
        self.n_dof = n_dof
        self.pair_idxs = list(combinations(range(16), r=2))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.labels_data is None:
            self.labels_data = h5py.File(os.path.join(self.root_dir, 'complete_data.hdf5'), 'r')

        obj_idx = int(idx / len(self.pair_idxs))
        pair_idx = self.pair_idxs[idx % len(self.pair_idxs)]

        # One Sample for us corresponds to one instantiation of an object type
        obj_data = self.labels_data['obj_' + str(obj_idx).zfill(6)]

        # Load depth image
        depth_imgs = torch.from_numpy(np.concatenate((obj_data['depth_imgs'],
                                                      np.expand_dims(obj_data['depth_imgs'][pair_idx[0]], axis=0),
                                                      np.expand_dims(obj_data['depth_imgs'][pair_idx[1]], axis=0))))

        depth_imgs.unsqueeze_(1).float()
        depth_imgs = torch.cat((depth_imgs, depth_imgs, depth_imgs), dim=1)

        # Load labels
        pt1 = obj_data['moving_frame_in_world'][pair_idx[0], :]
        pt2 = obj_data['moving_frame_in_world'][pair_idx[1], :]
        pt1_T_pt2 = change_frames(pt1, pt2)

        # Generating labels in screw notation: label := <l_hat, m, theta, d> = <3, 3, 1, 1>
        l_hat, m, theta, d = transform_to_screw(translation=pt1_T_pt2[:3],
                                                quat_in_wxyz=pt1_T_pt2[3:])

        label = torch.from_numpy(np.concatenate((l_hat, m, [theta], [d]))).float()
        sample = {'depth': depth_imgs,
                  'label': label}

        return sample


## Rigid Transform
class RigidTransformDataset(Dataset):
    def __init__(self,
                 ntrain,
                 root_dir,
                 n_dof):
        super(RigidTransformDataset, self).__init__()

        self.root_dir = root_dir
        self.labels_data = None
        self.length = ntrain
        self.n_dof = n_dof
        self.perm_idxs = list(permutations(range(16), r=2))

    def __len__(self):
        return self.length

    def __getitem__(self, idx, imgs_per_object=16):
        if self.labels_data is None:
            self.labels_data = h5py.File(os.path.join(self.root_dir, 'complete_data.hdf5'), 'r')

        obj_idx = int(idx / 240)
        obj_data_idx = self.perm_idxs[idx % 240]
        obj_data = self.labels_data['obj_' + str(obj_idx).zfill(6)]

        # Load depth image
        depth_imgs = torch.tensor([obj_data['depth_imgs'][obj_data_idx[0]],
                                   obj_data['depth_imgs'][obj_data_idx[1]]])
        depth_imgs.unsqueeze_(1).float()
        depth_imgs = torch.cat((depth_imgs, depth_imgs, depth_imgs), dim=1)

        # # Load labels
        pt1 = obj_data['moving_frame_in_world'][obj_data_idx[0], :]
        pt2 = obj_data['moving_frame_in_world'][obj_data_idx[1], :]
        pt1_T_pt2 = change_frames(pt1, pt2)

        # Generating labels in screw notation: label := <l_hat, m, theta, d> = <3, 3, 1, 1>
        l_hat, m, theta, d = transform_to_screw(translation=pt1_T_pt2[:3],
                                                quat_in_wxyz=pt1_T_pt2[3:])

        label = torch.from_numpy(np.concatenate((l_hat, m, [theta], [d]))).float()
        sample = {'depth': depth_imgs,
                  'label': label}

        return sample
