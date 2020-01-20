import numpy as np
import torch
import transforms3d as tf3d
import dq3d


def dual_quaternion_to_vecQuat_form(dq):
    trans = dq.translation()
    quat = dq.rotation().data  # expect x,y,z,w
    return np.concatenate((trans, quat))


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
        'screw_axis_std': (torch.std(l_hat_array, dim=0).cpu(),
                           torch.std(m_array, dim=0).cpu()),
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
