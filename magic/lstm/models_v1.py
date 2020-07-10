import torch
import torch.nn as nn
import torch.nn.functional as F
from ArticulationModelLearning.magic.lstm.utils import distance_bw_plucker_lines, \
    orientation_difference_bw_plucker_lines, expand_labels, theta_config_error, d_config_error
from torchvision import models


class DeepArtModel_v1(nn.Module):
    def __init__(self, lstm_hidden_dim=1000, n_lstm_hidden_layers=1, drop_p=0.5, n_output=8):
        super(DeepArtModel_v1, self).__init__()

        self.fc_res_dim_1 = 512
        self.lstm_input_dim = 1000
        self.lstm_hidden_dim = lstm_hidden_dim
        self.n_lstm_hidden_layers = n_lstm_hidden_layers
        self.fc_lstm_dim_1 = 256
        self.fc_lstm_dim_2 = 128
        self.n_output = n_output
        self.drop_p = drop_p

        self.resnet = models.resnet18()
        self.fc_res_1 = nn.Linear(self.lstm_input_dim, self.fc_res_dim_1)
        self.bn_res_1 = nn.BatchNorm1d(self.fc_res_dim_1, momentum=0.01)
        self.fc_res_2 = nn.Linear(self.fc_res_dim_1, self.lstm_input_dim)

        self.LSTM = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.n_lstm_hidden_layers,
            batch_first=True,
        )

        self.fc_lstm_1 = nn.Linear(self.lstm_hidden_dim, self.fc_lstm_dim_1)
        self.bn_lstm_1 = nn.BatchNorm1d(self.fc_lstm_dim_1, momentum=0.01)
        self.fc_lstm_2 = nn.Linear(self.fc_lstm_dim_1, self.fc_lstm_dim_2)
        self.bn_lstm_2 = nn.BatchNorm1d(self.fc_lstm_dim_2, momentum=0.01)
        self.dropout_layer1 = nn.Dropout(p=self.drop_p)
        self.fc_lstm_3 = nn.Linear(self.fc_lstm_dim_2, self.n_output)

        # # Initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        #         m.bias.data.fill_(0.01)
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.LSTM):
        #         for name, param in m.named_parameters():
        #             if 'bias' in name:
        #                 nn.init.constant_(param, 0.0)
        #             elif 'weight' in name:
        #                 nn.init.xavier_normal_(param)

    def forward(self, X_3d):
        # X shape: Batch x Sequence x 3 Channels x img_dims
        # Run resnet sequentially on the data to generate embedding sequence
        cnn_embed_seq = []
        for t in range(X_3d.size(1)):
            x = self.resnet(X_3d[:, t, :, :, :])
            x = x.view(x.size(0), -1)
            x = self.bn_res_1(self.fc_res_1(x))
            x = F.relu(x)
            x = self.fc_res_2(x)
            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

        # run lstm on the embedding sequence
        self.LSTM.flatten_parameters()

        RNN_out, (h_n, h_c) = self.LSTM(cnn_embed_seq, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) 
        None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x_rnn = RNN_out.contiguous().view(-1, self.lstm_hidden_dim)  # Using Last layer of RNN
        x_rnn = self.bn_lstm_1(self.fc_lstm_1(x_rnn))
        x_rnn = F.relu(x_rnn)
        x_rnn = self.bn_lstm_2(self.fc_lstm_2(x_rnn))
        x_rnn = F.relu(x_rnn)
        x_rnn = self.fc_lstm_3(x_rnn)
        return x_rnn.view(X_3d.size(0), -1)


def articulation_lstm_loss_spatial_distance_v1(pred, target, wt_on_ortho=1.):
    """ Based on Spatial distance
        Input shapes: Batch X Objects X images
    """
    pred = pred.view(pred.size(0), -1, 8)[:, 1:, :]  # We don't need the first row as it is for single image
    # pred = expand_labels(pred)  # Adding 3rd dimension to m, if needed

    # Spatial Distance loss
    dist_err = orientation_difference_bw_plucker_lines(target, pred) ** 2 + \
               distance_bw_plucker_lines(target, pred) ** 2

    # Configuration Loss
    # conf_err = ((target[:, :, 6:].clone() - pred[:, :, 6:].clone()) ** 2).sum(dim=-1)
    conf_err = theta_config_error(target, pred)**2 + d_config_error(target, pred)**2

    err = dist_err + conf_err
    loss = torch.mean(err)

    # print("Orientation error: ", ori_error.mean())
    # print("Distance_error: ", dist_error.mean())
    # print("Theta error: ", conf_error_theta.mean())
    # print("d error: ", conf_error_d.mean())

    # Ensure l_hat has norm 1.
    loss += torch.mean((torch.norm(pred[:, :, :3], dim=-1) - 1.) ** 2)

    # Ensure orthogonality between l_hat and m
    loss += wt_on_ortho * torch.mean(torch.abs(torch.sum(torch.mul(pred[:, :, :3], pred[:, :, 3:6]), dim=-1)))

    if torch.isnan(loss):
        print("target: Min: {},  Max{}".format(target.min(), target.max()))
        print("Prediction: Min: {},  Max{}".format(pred.min(), pred.max()))
        print("L2 error: {}".format(torch.mean((target - pred) ** 2)))
        print("Distance loss:{}".format(torch.mean(orientation_difference_bw_plucker_lines(target, pred) ** 2)))
        print("Orientation loss:{}".format(torch.mean(distance_bw_plucker_lines(target, pred) ** 2)))
        print("Configuration loss:{}".format(torch.mean(conf_err)))

    return loss


# class DeepArtModel_v2(nn.Module):
#     def __init__(self, lstm_hidden_dim=1000, n_lstm_hidden_layers=1, n_output=4):
#         super(DeepArtModel_v2, self).__init__()
#
#         self.lstm_input_dim = 1000
#         self.lstm_hidden_dim = lstm_hidden_dim
#         self.n_lstm_hidden_layers = n_lstm_hidden_layers
#         self.fc_lstm_dim_1 = 256
#         self.n_output = n_output
#
#         self.resnet = models.resnet18()
#         self.bn_res_1 = nn.BatchNorm1d(self.lstm_hidden_dim, momentum=0.01)
#         self.LSTM = nn.LSTM(
#             input_size=self.lstm_input_dim,
#             hidden_size=self.lstm_hidden_dim,
#             num_layers=self.n_lstm_hidden_layers,
#             batch_first=True,
#         )
#         self.fc_lstm_1 = nn.Linear(self.lstm_hidden_dim, self.fc_lstm_dim_1)
#         self.fc_lstm_2 = nn.Linear(self.fc_lstm_dim_1, self.n_output)
#
#     def forward(self, X_3d):
#         # X shape: Batch x Sequence x 3 Channels x img_dims
#         # Run resnet sequentially on the data to generate embedding sequence
#         cnn_embed_seq = []
#         for t in range(X_3d.size(1)):
#             x = self.resnet(X_3d[:, t, :, :, :])
#             x = x.view(x.size(0), -1)
#             x = self.bn_res_1(x)
#             cnn_embed_seq.append(x)
#
#         # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
#         cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
#
#         # run lstm on the embedding sequence
#         self.LSTM.flatten_parameters()
#
#         RNN_out, (h_n, h_c) = self.LSTM(cnn_embed_seq, None)
#         """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size)
#         None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """
#
#         # FC layers
#         x_rnn = RNN_out.contiguous().view(-1, self.lstm_hidden_dim)  # Using Last layer of RNN
#         x_rnn = self.fc_lstm_1(x_rnn)
#         x_rnn = F.relu(x_rnn)
#         x_rnn = self.bn_lstm_1(x_rnn)
#         x_rnn = self.fc_lstm_2(x_rnn)
#         return x_rnn.view(X_3d.size(0), -1)
#
#
# def articulation_lstm_loss_L2v2(pred, target):
#     """ L2 loss"""
#     pred = pred.view(pred.size(0), -1, 8)[:, 1:, :]  # We don't need the first row as it is for single image
#     return torch.mean((pred - target) ** 2)
