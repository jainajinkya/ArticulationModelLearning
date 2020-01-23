import numpy as np
import torch
import transforms3d as tf3d
import dq3d


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
