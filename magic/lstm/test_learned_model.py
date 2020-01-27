import numpy as np
import torch
import argparse
import matplotlib

from ArticulationModelLearning.magic.lstm.dataset import ArticulationDataset
from ArticulationModelLearning.magic.lstm.models import KinematicLSTMv0
from ArticulationModelLearning.magic.lstm.utils import dual_quaternion_to_screw_batch_mode

matplotlib.use('Agg')
from matplotlib import pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model for articulated object dataset.")
    parser.add_argument('--model-dir', type=str, default='../../models/')
    parser.add_argument('--model-name', type=str, default='test_lstm')
    parser.add_argument('--test-dir', type=str, default='../../../data/test/microwave/')
    parser.add_argument('--output-dir', type=str, default='../../../data/test/microwave/')
    parser.add_argument('--ntest', type=int, default=1, help='number of test samples (n_object_instants)')
    parser.add_argument('--ndof', type=int, default=1, help='how many degrees of freedom in the object class?')
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    parser.add_argument('--nwork', type=int, default=8, help='num_workers')
    parser.add_argument('--device', type=int, default=0, help='cuda device')
    parser.add_argument('--dual-quat', action='store_true', default=False, help='Dual quaternion representation or not')
    args = parser.parse_args()

    testset = ArticulationDataset(args.ntest,
                                  args.test_dir,
                                  n_dof=args.ndof)

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch,
                                             shuffle=True, num_workers=args.nwork,
                                             pin_memory=True)

    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')

    # load model
    best_model = KinematicLSTMv0(lstm_hidden_dim=1000, n_lstm_hidden_layers=1, h_fc_dim=256, n_output=120)
    best_model.load_state_dict(torch.load(args.model_dir + args.model_name + '.net'))
    best_model.float().to(device)
    best_model.eval()

    all_l_hat_err = torch.empty(0)
    all_m_err = torch.empty(0)
    all_m_err_abs = torch.empty(0)
    all_q_err = torch.empty(0)
    all_d_err = torch.empty(0)
    all_l_hat_std = torch.empty(0)
    all_m_std = torch.empty(0)
    all_m_std_abs = torch.empty(0)
    all_q_std = torch.empty(0)
    all_d_std = torch.empty(0)

    with torch.no_grad():
        for X in testloader:
            depth, labels = X['depth'].to(device), X['label'].to(device)
            y_pred = best_model(depth)
            y_pred = y_pred.view(y_pred.size(0), -1, 8)

            if args.dual_quat:
                y_pred = dual_quaternion_to_screw_batch_mode(y_pred)
                labels = dual_quaternion_to_screw_batch_mode(labels)

            err = labels - y_pred
            #all_l_hat_err = torch.cat(
             #   (all_l_hat_err, torch.mean(torch.norm(err[:, :, :3], dim=-1), dim=-1).cpu()))
            #all_m_err = torch.cat((all_m_err, torch.mean(torch.norm(err[:, :, 3:6], dim=-1), dim=-1).cpu()))
            l_hat_std, l_hat_mean = torch.std_mean(torch.acos(torch.mul(labels[:,:,:3], y_pred[:, :, :3]).sum(dim=-1)/(torch.norm(labels[:,:,:3], dim=-1)*torch.norm(y_pred[:,:,:3], dim=-1))), dim=-1)
            all_l_hat_err = torch.cat((all_l_hat_err, l_hat_mean.cpu()))

            m_std, m_mean = torch.std_mean(torch.norm(labels[:, :, 3:6], dim=-1) - torch.norm(y_pred[:, :, 3:6], dim=-1) , dim=-1)
            m_std_abs, m_mean_abs = torch.std_mean(torch.abs(torch.norm(labels[:, :, 3:6], dim=-1) - torch.norm(y_pred[:, :, 3:6], dim=-1)) , dim=-1)
            all_m_err = torch.cat((all_m_err, m_mean.cpu()))
            all_m_err_abs = torch.cat((all_m_err_abs, m_mean_abs.cpu()))

            all_q_err = torch.cat((all_q_err, torch.mean(err[:, :, 6], dim=-1).cpu()))
            all_d_err = torch.cat((all_d_err, torch.mean(err[:, :, 7], dim=-1).cpu()))

            #all_l_hat_std = torch.cat(
            #    (all_l_hat_std, torch.std(torch.norm(err[:, :, :3], dim=-1), dim=-1).cpu()))
            #all_m_std = torch.cat((all_m_std, torch.std(torch.norm(err[:, :, 3:6], dim=-1), dim=-1).cpu()))
            
            all_l_hat_std = torch.cat((all_l_hat_std, l_hat_std.cpu()))
            all_m_std = torch.cat((all_m_std, m_std.cpu()))
            all_m_std_abs = torch.cat((all_m_std_abs, m_std_abs.cpu()))
            all_q_std = torch.cat((all_q_std, torch.std(err[:, :, 6], dim=-1).cpu()))
            all_d_std = torch.cat((all_d_std, torch.std(err[:, :, 7], dim=-1).cpu()))

    # Plot variation of screw axis
    x_axis = np.arange(all_l_hat_err.size(0))

    fig = plt.figure(1)
    plt.errorbar(x_axis, all_l_hat_err.numpy(), all_l_hat_std.numpy(), marker='o', mfc='blue', ms=4., capsize=3., capthick=1.)
    plt.xlabel("Test object number")
    plt.ylabel("Angle error (rad)")
    plt.title("Test error in screw axis orientation")
    plt.tight_layout()
    plt.savefig(args.output_dir + '/l_hat_err_angle.png')
    plt.close(fig)

    fig = plt.figure(2)
    plt.errorbar(x_axis, all_m_err.numpy(), all_m_std.numpy(), marker='o', ms=4, mfc='blue', capsize=3., capthick=1.)
    plt.xlabel("Test object number")
    plt.ylabel("Error")
    plt.title("Test error in norm(m)")
    plt.tight_layout()
    plt.savefig(args.output_dir + '/m_err_norm.png')
    plt.close(fig)

    fig = plt.figure(2)
    plt.errorbar(x_axis, all_m_err_abs.numpy(), all_m_std_abs.numpy(), marker='o', mfc='blue', ms=4, capsize=3., capthick=1.)
    plt.xlabel("Test object number")
    plt.ylabel("Absolute Error")
    plt.title("Absolute test error in norm(m)")
    plt.tight_layout()
    plt.savefig(args.output_dir + '/m_err_abs_norm.png')
    plt.close(fig)

    fig = plt.figure(3)
    plt.errorbar(x_axis, all_q_err.numpy(), all_q_std.numpy(),  capsize=3., capthick=1.)
    plt.xlabel("Test object number")
    plt.ylabel("Error")
    plt.title("Test error in theta")
    plt.tight_layout()
    plt.savefig(args.output_dir + '/theta_err.png')
    plt.close(fig)

    fig = plt.figure(4)
    plt.errorbar(x_axis, all_d_err.numpy(), all_d_std.numpy(), capsize=3., capthick=1.)
    plt.xlabel("Test object number")
    plt.ylabel("Error")
    plt.title("Test error in d")
    plt.tight_layout()
    plt.savefig(args.output_dir + '/d_err.png')
    plt.close(fig)