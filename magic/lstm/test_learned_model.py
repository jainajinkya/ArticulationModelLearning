import argparse
import os

import GeneralizingKinematics.magic.mixture.mdn as mdn
import matplotlib
import numpy as np
import torch
from ArticulationModelLearning.magic.lstm.dataset import ArticulationDataset
from ArticulationModelLearning.magic.lstm.models import DeepArtModel
from ArticulationModelLearning.magic.lstm.utils import distance_bw_plucker_lines
from GeneralizingKinematics.magic.mixture.dataset import MixtureDataset
from GeneralizingKinematics.magic.mixture.models import KinematicMDNv3
from GeneralizingKinematics.magic.mixture.utils import *

matplotlib.use('Agg')
from matplotlib import pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model for articulated object dataset.")
    parser.add_argument('--model-dir', type=str, default='models/')
    parser.add_argument('--model-name', type=str, default='test_lstm')
    parser.add_argument('--test-dir', type=str, default='../data/test/microwave/')
    parser.add_argument('--output-dir', type=str, default='./plots/')
    parser.add_argument('--ntest', type=int, default=1, help='number of test samples (n_object_instants)')
    parser.add_argument('--aug-multi', type=int, default=120, help='Multiplier for data augmentation')
    parser.add_argument('--ndof', type=int, default=1, help='how many degrees of freedom in the object class?')
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    parser.add_argument('--nwork', type=int, default=8, help='num_workers')
    parser.add_argument('--device', type=int, default=0, help='cuda device')
    parser.add_argument('--dual-quat', action='store_true', default=False, help='Dual quaternion representation or not')
    parser.add_argument('--model-type', type=str, default='ours', help='ours, ben, li')
    parser.add_argument('--load-wts', action='store_true', default=False, help='Should load model wts from prior run?')
    parser.add_argument('--obj', type=str, default='microwave')

    args = parser.parse_args()

    ntest = args.ntest * args.aug_multi

    # Dataset for common axis
    all_ori_err_mean = torch.empty(0)
    all_ori_err_std = torch.empty(0)
    all_dist_err_mean = torch.empty(0)
    all_dist_err_std = torch.empty(0)

    output_dir = os.path.join(os.path.abspath(args.output_dir), args.model_type, args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')

    if args.model_type == 'ben':
        print("Testing Model: Ben et al.")
        bounds = np.load(os.path.join(args.test_dir, 'bounds.npy'))
        keep_columns = np.load(os.path.abspath('../GeneralizingKinematics/keep_columns_' + args.obj + '.npy'))
        one_columns = np.load(os.path.abspath('../GeneralizingKinematics/one_columns_' + args.obj + '.npy'))
        testset = MixtureDataset(ntest,
                                 args.test_dir,
                                 n_dof=args.ndof,
                                 normalize=True,
                                 bounds=bounds,
                                 keep_columns=keep_columns,
                                 one_columns=one_columns
                                 )

        best_model = KinematicMDNv3(n_gaussians=20,
                                    out_features=testset.labels.shape[1])
        best_model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name + '.net')))
        best_model.float().to(device)
        best_model.eval()

        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch,
                                                 shuffle=False, num_workers=args.nwork,
                                                 pin_memory=True)
        with torch.no_grad():
            for X in testloader:
                depth, labels = X['depth'].to(device), X['label'].to(device)
                pi, sigma, mu = best_model(depth)

        output = mdn.sample(pi, sigma, mu)

        # expand labels and output back to full dataset size
        output = expand_labels(testloader.dataset.full_labels[:len(labels)],
                               output,
                               testloader.dataset.keep_columns,
                               testloader.dataset.one_columns)

        labels = expand_labels(testloader.dataset.full_labels[:len(labels)],
                               labels,
                               testloader.dataset.keep_columns,
                               testloader.dataset.one_columns)

        # interpret and convert to real.
        ground_truth_params = convert_dict_to_real(interpret_labels(labels, args.ndof), bounds, args.ndof)
        param_dict = convert_dict_to_real(interpret_labels(output, args.ndof), bounds, args.ndof)

        real_axis = ground_truth_params['axis']
        real_net_axis = param_dict['axis']

        import pdb; pdb.set_trace()
        all_dist_err_std, all_dist_err_mean = torch.std_mean(
            torch.norm(real_axis[:, :, :3] - real_net_axis[:, :, :3], dim=-1, keepdim=True), dim=-1)
        all_dist_err_std = all_dist_err_std.squeeze_(dim=-1).cpu()
        all_dist_err_mean = all_dist_err_mean.squeeze_(dim=-1).cpu()

        all_ori_err_std, all_ori_err_mean = torch.std_mean(
            difference_between_quaternions_tensors(real_axis[:, :, 3:7], real_net_axis[:, :, 3:7]), dim=-1)
        all_ori_err_mean = all_dist_err_mean.cpu()
        all_ori_err_std = all_ori_err_std.cpu()

        # Configuration error
        real_configs = ground_truth_params['config']
        net_configs = param_dict['config']

        all_q_err_std, all_q_err_mean = torch.std_mean(
            torch.norm(real_configs[:, :, :] - net_configs[:, :, :], dim=-1, keepdim=True), dim=-1)
        all_q_err_std = all_q_err_std.squeeze_(dim=-1).cpu().numpy()
        all_q_err_mean = all_q_err_mean.squeeze_(dim=-1).cpu().numpy()

        x_axis = np.arange(np.shape(all_q_err_mean)[0])
        fig = plt.figure(3)
        plt.errorbar(x_axis, all_q_err_mean, all_q_err_std, capsize=3., capthick=1., ls='none')
        plt.xlabel("Test object number")
        plt.ylabel("Error in Config")
        plt.title("Test error in Configurations")
        plt.tight_layout()
        plt.savefig(output_dir + '/theta_err.png')
        plt.close(fig)

    elif args.model_type == 'li':
        print("Testing Model: Li et al.")

    else:
        print("Testing Model: Ours")

        # load model
        # best_model = KinematicLSTMv0(lstm_hidden_dim=1000, n_lstm_hidden_layers=1, h_fc_dim=256, n_output=8)
        best_model = DeepArtModel(lstm_hidden_dim=1000, n_lstm_hidden_layers=1, h_fc_dim=256, n_output=8)
        best_model.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name + '.net')))
        best_model.float().to(device)
        best_model.eval()

        testset = ArticulationDataset(ntest,
                                      args.test_dir,
                                      n_dof=args.ndof)

        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch,
                                                 shuffle=False, num_workers=args.nwork,
                                                 pin_memory=True)

        all_q_err_mean = torch.empty(0)
        all_q_err_std = torch.empty(0)
        all_d_err_mean = torch.empty(0)
        all_d_err_std = torch.empty(0)

        obj_idxs = torch.empty(0)  # Recording object indexes for analysis

        # # Data collection for particle filter testing
        # all_labels = torch.empty(0)
        # all_preds = torch.empty(0)
        # all_errs = torch.empty(0)

        with torch.no_grad():
            for X in testloader:
                depth, labels = X['depth'].to(device), X['label'].to(device)
                y_pred = best_model(depth)
                y_pred = y_pred.view(y_pred.size(0), -1, 8)
                y_pred = y_pred[:, 1:, :]

                # Orientation error
                ori_err_std, ori_err_mean = torch.std_mean(torch.acos(
                    torch.mul(labels[:, :, :3], y_pred[:, :, :3]).sum(dim=-1) / (
                            torch.norm(labels[:, :, :3], dim=-1) * torch.norm(y_pred[:, :, :3], dim=-1))), dim=-1)
                all_ori_err_mean = torch.cat((all_ori_err_mean, ori_err_mean.cpu()))
                all_ori_err_std = torch.cat((all_ori_err_std, ori_err_std.cpu()))

                # Distance b/w plucker lines
                dist_err_std, dist_err_mean = torch.std_mean(distance_bw_plucker_lines(labels, y_pred), dim=-1)
                all_dist_err_mean = torch.cat((all_dist_err_mean, dist_err_mean.cpu()))
                all_dist_err_std = torch.cat((all_dist_err_std, dist_err_std.cpu()))

                # Configurational errors
                q_err_std, q_err_mean = torch.std_mean(
                    torch.norm(labels[:, :, 6] - y_pred[:, :, 6], dim=-1, keepdim=True), dim=-1)
                all_q_err_mean = torch.cat((all_q_err_mean, q_err_mean.cpu()))
                all_q_err_std = torch.cat((all_q_err_std, q_err_std.cpu()))

                d_err_std, d_err_mean = torch.std_mean(
                    torch.norm(labels[:, :, 7] - y_pred[:, :, 7], dim=-1, keepdim=True), dim=-1)
                all_d_err_mean = torch.cat((all_d_err_std, d_err_mean.cpu()))
                all_d_err_std = torch.cat((all_d_err_std, d_err_std.cpu()))

                # # Data for particle filter
                # all_labels = torch.cat((all_labels, labels.cpu()))
                # all_preds = torch.cat((all_preds, y_pred.cpu()))
                # all_errs = torch.cat((all_errs, (labels - y_pred).cpu()))

        # Sort objects as per the idxs
        x_axis = np.arange(all_q_err_mean.size(0))

        fig = plt.figure(3)
        plt.errorbar(x_axis, all_q_err_mean.numpy(), all_q_err_std.numpy(), capsize=3., capthick=1., ls='none')
        plt.xlabel("Test object number")
        plt.ylabel("Error (rad)")
        plt.title("Test error in theta")
        plt.tight_layout()
        plt.savefig(output_dir + '/theta_err.png')
        plt.close(fig)

        fig = plt.figure(4)
        plt.errorbar(x_axis, all_d_err_mean.numpy(), all_d_err_std.numpy(), capsize=3., capthick=1., ls='none')
        plt.xlabel("Test object number")
        plt.ylabel("Error (m)")
        plt.title("Test error in d")
        plt.tight_layout()
        plt.savefig(output_dir + '/d_err.png')
        plt.close(fig)

        # # Storing data for particle filter analysis
        # p_data = {'labels': all_labels.numpy(), 'predictions': all_preds.numpy(), 'errors': all_errs.numpy()}
        # import pickle
        # pickle.dump(p_data, open(output_dir + '/test_prediction_data.pkl', 'wb'))

    """ Common Plots"""
    # Plot variation of screw axis
    x_axis = np.arange(all_ori_err_mean.size(0))
    fig = plt.figure(1)
    plt.errorbar(x_axis, all_ori_err_mean.numpy(), all_ori_err_std.numpy(), marker='o', mfc='blue', ms=4., capsize=3.,
                 capthick=1., ls='none')
    plt.xlabel("Test object number")
    plt.ylabel("Orientation error (rad)")
    plt.title("Test error in screw axis orientation")
    plt.tight_layout()
    plt.savefig(output_dir + '/orientation_test_error.png')
    plt.close(fig)

    fig = plt.figure(2)
    plt.errorbar(x_axis, all_dist_err_mean.numpy(), all_dist_err_std.numpy(), marker='o', ms=4, mfc='blue', capsize=3.,
                 capthick=1., ls='none')
    plt.xlabel("Test object number")
    plt.ylabel("Spatial distance error (m)")
    plt.title("Test error in spatial distance")
    plt.tight_layout()
    plt.savefig(output_dir + '/distance_test_error.png')
    plt.close(fig)
