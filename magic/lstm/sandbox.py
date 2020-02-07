import argparse
import os
from itertools import permutations
import h5py
import torch
import numpy as np

from ArticulationModelLearning.magic.lstm.utils import transform_to_screw
from SyntheticArticulatedData.generation.utils import change_frames


def create_dataset_2_imgs(root_dir):
    input_dataset = h5py.File(os.path.join(root_dir, 'complete_data.hdf5'), 'r')
    depth_imgs = []
    labels = []
    idxs = list(permutations(range(16), r=2))

    for obj in input_dataset.keys():
        obj_data = input_dataset[obj]

        # Create pairs of images and labels
        for idx in idxs:
            depth_imgs.append(np.array([obj_data['depth_imgs'][idx[0]],
                                        obj_data['depth_imgs'][idx[1]]]))

            pt1 = obj_data['moving_frame_in_world'][idx[0], :]
            pt2 = obj_data['moving_frame_in_world'][idx[1], :]
            pt1_T_pt2 = change_frames(pt1, pt2)

            # Generating labels in screw notation: label := <l_hat, m, theta, d> = <3, 3, 1, 1>
            l_hat, m, theta, d = transform_to_screw(translation=pt1_T_pt2[:3],
                                                    quat_in_wxyz=pt1_T_pt2[3:])

            labels.append(np.concatenate((l_hat, m, [theta], [d])))

    h5File = h5py.File(root_dir + 'complete_data_2_imgs.hdf5', 'w')
    h5File.create_dataset('depth_imgs', data=depth_imgs)
    h5File.create_dataset('labels', data=labels)
    print("Converted input data to a dataset having 2 images as depth images. "
          "Dataset saved at: {}".format(root_dir + 'complete_data_2_imgs.hdf5'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str)
    args = parser.parse_args()

    create_dataset_2_imgs(args.dataset_dir)