import os
from itertools import combinations

import h5py
import numpy as np
import torch
from ArticulationModelLearning.magic.lstm.utils import transform_to_screw, change_frames, transform_plucker_line
from torch.utils.data import Dataset


class ArticulationDataset(Dataset):
    def __init__(self,
                 ntrain,
                 root_dir,
                 n_dof,
                 norm_factor=1.,
                 transform=None):
        super(ArticulationDataset, self).__init__()

        self.root_dir = root_dir
        self.labels_data = None
        self.length = ntrain
        self.n_dof = n_dof
        self.normalization_factor = norm_factor
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.labels_data is None:
            self.labels_data = h5py.File(os.path.join(self.root_dir, 'complete_data.hdf5'), 'r')

        # One Sample for us corresponds to one instantiation of an object type
        obj_data = self.labels_data['obj_' + str(idx).zfill(6)]

        # Load depth image
        depth_imgs = torch.tensor(obj_data['depth_imgs'])

        if self.transform is not None:
            depth_imgs = self.transform(depth_imgs)

        depth_imgs.unsqueeze_(1).float()

        depth_imgs = torch.cat((depth_imgs, depth_imgs, depth_imgs), dim=1)

        # Load labels
        moving_body_poses = obj_data['moving_frame_in_world']
        label = np.empty((len(moving_body_poses) - 1, 8))

        # Object pose in world
        obj_pose_in_world = np.array(obj_data['embedding_and_params'])[-7:]  # obj_pose, obj_quat_wxyz
        pt1 = moving_body_poses[0, :]  # Fixed common reference frame
        obj_T_pt1 = change_frames(obj_pose_in_world, pt1)

        for i in range(len(moving_body_poses) - 1):
            pt2 = moving_body_poses[i + 1, :]
            pt1_T_pt2 = change_frames(pt1, pt2)

            # Generating labels in screw notation: label := <l_hat, m, theta, d> = <3, 3, 1, 1>
            l_hat, m, theta, d = transform_to_screw(translation=pt1_T_pt2[:3],
                                                    quat_in_wxyz=pt1_T_pt2[3:])

            # label[i, :] = np.concatenate((l_hat, m, [theta], [d]))  # This defines frames wrt pt 1

            # Convert line in object_local_coordinates
            new_l = transform_plucker_line(np.concatenate((l_hat, m)), trans=obj_T_pt1[:3], quat=obj_T_pt1[3:])
            label[i, :] = np.concatenate((new_l, [theta], [d]))  # This defines frames wrt pt 1

        # Normalize labels
        label[:, 3:6] /= self.normalization_factor

        label = torch.from_numpy(label).float()
        sample = {'depth': depth_imgs,
                  'label': label}

        return sample


#
# ### LSTM with Data Augmentation
# class ArticulationDatasetV2(Dataset):
#     def __init__(self,
#                  ntrain,
#                  root_dir,
#                  n_dof):
#         super(ArticulationDatasetV2, self).__init__()
#
#         self.root_dir = root_dir
#         self.labels_data = None
#         self.length = ntrain
#         self.n_dof = n_dof
#         self.pair_idxs = all_combinations(n=8)  # no. of total images available = 16
#         self.img_size = (108, 192)
#
#     def find_dense_idx(self, s_idx):
#         return 2*s_idx
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, idx):
#         if self.labels_data is None:
#             self.labels_data = h5py.File(os.path.join(self.root_dir, 'complete_data.hdf5'), 'r')
#
#         obj_idx = int(idx / len(self.pair_idxs))
#         pair_idx = self.pair_idxs[idx % len(self.pair_idxs)]
#
#         # Convert to dense idxs
#         pair_idx = [self.find_dense_idx(idx) for idx in pair_idx]
#
#         # One Sample for us corresponds to one instantiation of an object type
#         obj_data = self.labels_data['obj_' + str(obj_idx).zfill(6)]
#
#         # Load depth images
#         depth_imgs = torch.empty(size=(len(pair_idx), self.img_size[0], self.img_size[1]))
#         for i, idx in enumerate(pair_idx):
#             depth_imgs[i, :, :] = torch.from_numpy(obj_data['depth_imgs'][idx])
#
#         depth_imgs.unsqueeze_(1).float()
#         depth_imgs = torch.cat((depth_imgs, depth_imgs, depth_imgs), dim=1)
#
#         # Load labels
#         label = []
#
#         for i in range(len(pair_idx)-1):
#             pt1 = obj_data['moving_frame_in_world'][pair_idx[i], :]
#             pt2 = obj_data['moving_frame_in_world'][pair_idx[i+1], :]
#             pt1_T_pt2 = change_frames(pt1, pt2)
#
#             # Generating labels in screw notation: label := <l_hat, m, theta, d> = <3, 3, 1, 1>
#             l_hat, m, theta, d = transform_to_screw(translation=pt1_T_pt2[:3],
#                                                     quat_in_wxyz=pt1_T_pt2[3:])
#
#             label.append(np.concatenate((l_hat, m, [theta], [d])))
#
#         label = torch.tensor(label)
#
#         sample = {'depth': depth_imgs,
#                   'label': label}
#
#         return sample


### LSTM + 2 images
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
        self.n_step_idxs = 4


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
        depth_imgs = torch.from_numpy(np.concatenate((obj_data['depth_imgs'][:-1:self.n_step_idxs],
                                                      np.expand_dims(obj_data['depth_imgs'][pair_idx[0]], axis=0),
                                                      np.expand_dims(obj_data['depth_imgs'][pair_idx[1]], axis=0))))

        depth_imgs.unsqueeze_(1).float()
        # depth_imgs = torch.cat((depth_imgs, depth_imgs, depth_imgs), dim=1)
        depth_imgs = depth_imgs.repeat(1, 3, 1, 1)

        # Load labels
        all_labels = torch.tensor(obj_data['all_transforms'][:-1:self.n_step_idxs]).float()

        # Query label
        pt1 = obj_data['moving_frame_in_world'][pair_idx[0], :]
        pt2 = obj_data['moving_frame_in_world'][pair_idx[1], :]
        pt1_T_pt2 = change_frames(pt1, pt2)

        # Generating labels in screw notation: label := <l_hat, m, theta, d> = <3, 3, 1, 1>
        l_hat, m, theta, d = transform_to_screw(translation=pt1_T_pt2[:3],
                                                quat_in_wxyz=pt1_T_pt2[3:])

        label = torch.from_numpy(np.concatenate((l_hat, m, [theta], [d]))).float()
        sample = {'depth': depth_imgs,
                  'all_labels': all_labels,
                  'label': label}

        return sample


## Rigid Transform
class RigidTransformDataset(Dataset):
    def __init__(self,
                 ntrain,
                 root_dir,
                 n_dof,
                 norm_factor=1.,
                 transform=None):
        super(RigidTransformDataset, self).__init__()

        self.root_dir = root_dir
        self.labels_data = None
        self.length = ntrain
        self.n_dof = n_dof
        self.normalization_factor = norm_factor
        self.transform = transform
        self.augmentation_factor = 15

    def __len__(self):
        return self.length

    def __getitem__(self, idx, imgs_per_object=16):
        if self.labels_data is None:
            self.labels_data = h5py.File(os.path.join(self.root_dir, 'complete_data.hdf5'), 'r')

        obj_idx = int(idx / self.augmentation_factor)
        obj_data_idx = idx % self.augmentation_factor + 1
        obj_data = self.labels_data['obj_' + str(obj_idx).zfill(6)]

        # Load depth image
        depth_imgs = torch.tensor([obj_data['depth_imgs'][0],
                                   obj_data['depth_imgs'][obj_data_idx]])
        depth_imgs.unsqueeze_(1).float()
        depth_imgs = torch.cat((depth_imgs, depth_imgs, depth_imgs), dim=1)

        # # Load labels
        pt1 = obj_data['moving_frame_in_world'][0, :]
        pt2 = obj_data['moving_frame_in_world'][obj_data_idx, :]
        pt1_T_pt2 = change_frames(pt1, pt2)

        # Object pose in world
        obj_pose_in_world = np.array(obj_data['embedding_and_params'])[-7:]  # obj_pose, obj_quat_wxyz
        obj_T_pt1 = change_frames(obj_pose_in_world, pt1)

        # Generating labels in screw notation: label := <l_hat, m, theta, d> = <3, 3, 1, 1>
        l_hat, m, theta, d = transform_to_screw(translation=pt1_T_pt2[:3],
                                                quat_in_wxyz=pt1_T_pt2[3:])
        # label = torch.from_numpy(np.concatenate((l_hat, m, [theta], [d]))).float()

        # Convert line in object_local_coordinates
        new_l = transform_plucker_line(np.concatenate((l_hat, m)), trans=obj_T_pt1[:3], quat=obj_T_pt1[3:])
        label = np.concatenate((new_l, [theta], [d]))  # This defines frames wrt pt 1

        # Normalize labels
        label[:, 3:6] /= self.normalization_factor

        label = torch.from_numpy(label).float()
        sample = {'depth': depth_imgs,
                  'label': label}

        return sample
