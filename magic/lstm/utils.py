from itertools import combinations

import dq3d
import h5py
import numpy as np
import torch
import transforms3d as tf3d


# Dual Quaternion utils
def dual_quaternion_to_vecQuat_form(dq):
    trans = dq.translation()
    quat = dq.rotation().data  # expect x,y,z,w
    return np.concatenate((trans, quat))


def transform_to_screw(translation, quat_in_wxyz, tol=1e-6):
    dq = dq3d.dualquat(dq3d.quat(quat_as_xyzw(quat_in_wxyz)), translation)
    screw = dual_quaternion_to_screw(dq, tol)
    return screw


def dual_quaternion_to_screw(dq, tol=1e-6):
    l_hat, theta = tf3d.quaternions.quat2axangle(np.array([dq.real.w, dq.real.x, dq.real.y, dq.real.z]))

    if theta < tol or abs(theta - np.pi) < tol:
        t_vec = dq.translation()
        l_hat = t_vec / np.linalg.norm(t_vec)
        theta = tol  # This makes sure that tan(theta) is defined
    else:
        t_vec = (2 * tf3d.quaternions.qmult(dq.dual.data, tf3d.quaternions.qconjugate(dq.real.data)))[
                1:]  # taking xyz from wxyz

    d = t_vec.dot(l_hat)
    m = (1 / 2) * (np.cross(t_vec, l_hat) + ((t_vec - d * l_hat) / np.tan(theta / 2)))
    return l_hat, m, theta, d


def dual_quaternion_to_screw_batch_mode(dq_batch):
    screws = torch.empty(0)
    for b in dq_batch:
        for dq in b:
            l_hat, m, theta, d = dual_quaternion_to_screw(tensor_to_dual_quat(dq))
            screws = torch.cat((screws, torch.from_numpy(np.concatenate((l_hat, m, [theta], [d]))).float()))
    return screws.view_as(dq_batch)


def tensor_to_dual_quat(dq):
    dq = dq.cpu().numpy()
    dq1 = dq3d.dualquat.zeros()
    dq1.real.data = quat_as_wxyz(dq[:4])
    dq1.dual.data = quat_as_wxyz(dq[4:])
    return dq1


# Generic Utils
def quat_as_wxyz(q):
    # Assume q in xyzw
    new_q = np.array([q[3], q[0], q[1], q[2]])
    return new_q / np.linalg.norm(new_q)


def quat_as_xyzw(q):
    # Assume q in wxyz
    new_q = np.array([q[1], q[2], q[3], q[0]])
    return new_q / np.linalg.norm(new_q)


def orientation_difference(q1, q2):
    # Assume q1 and q2 are in xyzw form
    # Normalize them
    rot1 = tf3d.quaternions.quat2mat(quat_as_wxyz(q1))
    rot2 = tf3d.quaternions.quat2mat(quat_as_wxyz(q2))

    should_be_eye = np.matmul(rot1.T, rot2)  # (rot1.inverse = rot1.T) * (rot2)
    I_ = np.eye(3)
    return np.linalg.norm(I_ - should_be_eye, ord='fro')


def pose_difference(f1, f2, pos_wt=1.0, ori_wt=1.0, dual_quats=False):
    # Input: 7x1 , first 3 corresponds to the origin and last 4 orientation quaternion
    position_diff = np.linalg.norm(f1[:3] - f2[:3])
    return pos_wt * position_diff + ori_wt * orientation_difference(f1[3:], f2[3:])


def detect_model_class(mv_frames):
    model_class_name = 'revolute'
    return model_class_name


def interpret_label(label):
    label = label.view(-1, 8)
    l_hat_array = label[:, :3]
    m_array = label[:, 3:6]
    q_array = label[:, 6]
    d_array = label[:, 7]

    return {
        'screw_axis': (torch.mean(l_hat_array, dim=0).cpu(),
                       torch.mean(m_array, dim=0).cpu()),
        'l_hat_array': l_hat_array.cpu(),
        'm_array': m_array.cpu(),
        'theta_array': q_array.cpu(),
        'd_array': d_array.cpu()
    }

    # # A single label consists of reference frame dual quaternion and moving frame quats
    # ref_dq = label[0, :]
    # mv_dqs = label[1:, :]
    #
    # ref_frame = dual_quaternion_to_vecQuat_form(ref_dq)
    # mv_frames = []
    # for dq in mv_dqs:
    #     mv_frames.append(dual_quaternion_to_vecQuat_form(dq))
    #
    # return {
    #     'reference_frame': ref_frame,
    #     'moving_frames': mv_frames
    # }
    # # Interpret model parameters
    # model_class, params, configs = detect_model_class(mv_frames)
    #
    # # Returns model class, reference frame, configurations, other model parameters
    # return {'model_class': model_class,
    #         'reference_frame': ref_frame,
    #         'configs': configs,
    #         'params': params}


def all_combinations(n):
    idxs = []
    for r in np.arange(2, n + 1):
        idxs.append(list(combinations(range(n), r=r)))
    return [item for sublist in idxs for item in sublist]


def find_all_labels(moving_body_poses):
    all_labels = np.empty((len(moving_body_poses), 8))
    for i in range(len(moving_body_poses) - 1):
        pt1 = moving_body_poses[i, :]
        pt2 = moving_body_poses[i + 1, :]
        pt1_T_pt2 = change_frames(pt1, pt2)
        l_hat, m, theta, d = transform_to_screw(translation=pt1_T_pt2[:3],
                                                quat_in_wxyz=pt1_T_pt2[3:])
        all_labels[i + 1, :] = np.concatenate((l_hat, m, [theta], [d]))

    # Adding zeros for first image as padding for correct shapes
    all_labels[0, :] = np.concatenate((all_labels[1, :6], [0.], [0.]))
    return all_labels


def append_all_labels_to_dataset(filename):
    all_data = h5py.File(filename, 'r+')
    for key in all_data.keys():
        obj_data = all_data[key]
        moving_body_poses = obj_data['moving_frame_in_world']
        obj_data['all_transforms'] = find_all_labels(moving_body_poses)
    all_data.close()
    print("Added all transforms to the dataset.")


def distance_bw_plucker_lines(target, prediction, eps=1e-10):
    """ Input shapes Tensors: Batch X #Images X 8
    # Based on formula from Plücker Coordinates for Lines in the Space by Prof. Yan-bin Jia
    # Verified by https://keisan.casio.com/exec/system/1223531414
    """
    norm_cross_prod = torch.norm(torch.cross(target[:, :, :3], prediction[:, :, :3], dim=-1), dim=-1)
    dist = torch.zeros_like(norm_cross_prod)

    # Checking for Parallel Lines
    if torch.any(norm_cross_prod <= eps):
        zero_idxs = (norm_cross_prod <= eps).nonzero(as_tuple=True)
        scales = torch.norm(prediction[zero_idxs][:, :3], dim=-1) / torch.norm(target[zero_idxs][:, :3], dim=-1) + eps
        dist[zero_idxs] = torch.norm(torch.cross(target[zero_idxs][:, :3], (
                target[zero_idxs][:, 3:6] - prediction[zero_idxs][:, 3:6] / scales.unsqueeze(-1))), dim=-1) / (
                                  torch.mul(target[zero_idxs][:, :3], target[zero_idxs][:, :3]).sum(dim=-1) + eps)

    # Skew Lines: Non zero cross product
    nonzero_idxs = (norm_cross_prod > eps).nonzero(as_tuple=True)
    dist[nonzero_idxs] = torch.abs(
        torch.mul(target[nonzero_idxs][:, :3], prediction[nonzero_idxs][:, 3:6]).sum(dim=-1) + torch.mul(
            target[nonzero_idxs][:, 3:6], prediction[nonzero_idxs][:, :3]).sum(dim=-1)) / (
                                 norm_cross_prod[nonzero_idxs] + eps)
    return dist


def orientation_difference_bw_plucker_lines(target, prediction, eps=1e-6):
    """ Input shapes Tensors: Batch X #Images X 8
    range of arccos ins [0, pi)"""
    return torch.acos(torch.clamp(torch.mul(target[:, :, :3], prediction[:, :, :3]).sum(dim=-1) / (
            torch.norm(target[:, :, :3], dim=-1) * torch.norm(prediction[:, :, :3], dim=-1) + eps),
                                  min=-1, max=1))


def quaternion_inner_product(q, r):
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    original_shape = q.shape
    return torch.bmm(q.view(-1, 1, 4), r.view(-1, 4, 1)).view(original_shape[:-1])


def difference_between_quaternions_tensors(q1, q2, eps=1e-6):
    return torch.acos(torch.clamp(2 * quaternion_inner_product(q1, q2) ** 2 - 1, -1 + eps, 1 - eps))


def transform_plucker_line(line, trans, quat):
    trasform = np.zeros(6, 6)
    rot_mat = tf3d.quaternions.quat2mat(quat)
    t_mat = to_skew_symmetric_matrix(trans)
    trasform[0:3, 3:6] = rot_mat
    trasform[3:6, 0:3] = rot_mat
    trasform[3:6, 3:6] = np.matmul(t_mat, rot_mat)
    return np.matmul(trasform, line)


def to_skew_symmetric_matrix(v):
    return np.array([[0., -v[2], v[1]],
                     [v[2], 0., -v[0]],
                     [-v[1], v[0], 0.]])


def transform_plucker_line_batch(line, trans, quat):
    # line : <l_hat, m>
    transform = torch.zeros((6, 6))
    rot_mat = torch.from_numpy(tf3d.quaternions.quat2mat(quat))
    t_mat = to_skew_symmetric_matrix_batch(trans)
    transform[0:3, 3:6] = rot_mat
    transform[3:6, 0:3] = rot_mat
    transform[3:6, 3:6] = torch.matmul(t_mat, rot_mat)
    return torch.matmul(transform, line.transpose(-1, -2)).transpose(-1, -2)


def to_skew_symmetric_matrix_batch(vec):
    assert vec.shape[-1] == 3
    original_shape = vec.size()
    vec = vec.view(-1, 3)
    skew_mat = torch.zeros((vec.size(0), 3, 3))
    skew_mat[:, 0, 1] = -vec[:, 2]
    skew_mat[:, 0, 2] = vec[:, 1]
    skew_mat[:, 1, 0] = vec[:, 2]
    skew_mat[:, 1, 2] = -vec[:, 0]
    skew_mat[:, 2, 0] = -vec[:, 1]
    skew_mat[:, 2, 1] = vec[:, 0]
    return skew_mat.view(list(original_shape[:-1]) + [3, 3])


def change_frames(frame_B_wrt_A, pose_wrt_A):
    A_T_pose = tf3d.affines.compose(T=pose_wrt_A[:3],
                                    R=tf3d.quaternions.quat2mat(pose_wrt_A[3:]),  # quat in  wxyz
                                    Z=np.ones(3))
    A_rot_mat_B = tf3d.quaternions.quat2mat(frame_B_wrt_A[3:])

    # Following as described in Craig
    B_T_A = tf3d.affines.compose(T=-A_rot_mat_B.T.dot(frame_B_wrt_A[:3]),
                                 R=A_rot_mat_B.T,
                                 Z=np.ones(3))

    B_T_pose = B_T_A.dot(A_T_pose)
    trans, rot, scale, _ = tf3d.affines.decompose44(B_T_pose)
    quat = tf3d.quaternions.mat2quat(rot)
    return np.concatenate((trans, quat))  # return quat in wxyz

# Plotting Utils
def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)
