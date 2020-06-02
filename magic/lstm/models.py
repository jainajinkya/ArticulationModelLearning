import torch
import torch.nn as nn
import torch.nn.functional as F
from ArticulationModelLearning.magic.lstm.utils import distance_bw_plucker_lines, \
    orientation_difference_bw_plucker_lines
from torchvision import models


# v1
class DeepArtModel(nn.Module):
    def __init__(self, lstm_hidden_dim=1000, n_lstm_hidden_layers=1, drop_p=0.5,
                 h_fc_dim=256, n_output=8):
        super(DeepArtModel, self).__init__()

        self.fc_res_dim_1 = 1000
        self.fc_res_dim_2 = 512
        self.lstm_input_dim = 1000
        self.lstm_hidden_dim = lstm_hidden_dim
        self.n_lstm_hidden_layers = n_lstm_hidden_layers
        self.fc_lstm_dim_1 = 512
        self.fc_lstm_dim_2 = 256
        self.n_output = n_output
        self.drop_p = drop_p

        self.resnet = models.resnet18()
        self.fc_res_1 = nn.Linear(self.lstm_input_dim, self.fc_res_dim_1)
        self.bn_res_1 = nn.BatchNorm1d(self.fc_res_dim_1, momentum=0.01)
        self.fc_res_2 = nn.Linear(self.fc_res_dim_1, self.fc_res_dim_2)
        self.bn_res_2 = nn.BatchNorm1d(self.fc_res_dim_2, momentum=0.01)
        self.fc_res_3 = nn.Linear(self.fc_res_dim_2, self.lstm_input_dim)

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

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.xavier_normal_(param)

    def forward(self, X_3d):
        # X shape: Batch x Sequence x 3 Channels x img_dims
        # Run resnet sequentially on the data to generate embedding sequence
        cnn_embed_seq = []
        for t in range(X_3d.size(1)):
            x = self.resnet(X_3d[:, t, :, :, :])
            x = x.view(x.size(0), -1)
            x = self.bn_res_1(self.fc_res_1(x))
            x = F.relu(x)
            x = self.bn_res_2(self.fc_res_2(x))
            x = F.relu(x)
            x = self.fc_res_3(x)
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
        x_rnn = self.dropout_layer1(x_rnn)
        x_rnn = self.fc_lstm_3(x_rnn)
        return x_rnn.view(X_3d.size(0), -1)


# v0
class KinematicLSTMv0(nn.Module):
    def __init__(self, lstm_hidden_dim=1000, n_lstm_hidden_layers=1, drop_p=0.5,
                 h_fc_dim=256, n_output=8):
        super(KinematicLSTMv0, self).__init__()

        self.lstm_input_dim = 1000
        self.lstm_hidden_dim = lstm_hidden_dim
        self.n_lstm_hidden_layers = n_lstm_hidden_layers
        self.drop_p = drop_p
        self.h_fc_dim = h_fc_dim
        self.n_output = n_output

        self.resnet = models.resnet18()
        self.fc1 = nn.Linear(self.lstm_input_dim, self.lstm_input_dim)
        self.bn1 = nn.BatchNorm1d(self.lstm_input_dim, momentum=0.01)

        self.LSTM = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.n_lstm_hidden_layers,
            batch_first=True,
        )

        self.fc2 = nn.Linear(self.lstm_hidden_dim, self.h_fc_dim)
        self.bn2 = nn.BatchNorm1d(self.h_fc_dim, momentum=0.01)
        self.dropout_layer1 = nn.Dropout(p=self.drop_p)
        self.fc3 = nn.Linear(self.h_fc_dim, self.n_output)

    def forward(self, X_3d):
        # X shape: Batch x Sequence x 3 Channels x img_dims
        # Run resnet sequentially on the data to generate embedding sequence
        cnn_embed_seq = []
        for t in range(X_3d.size(1)):
            x = self.resnet(X_3d[:, t, :, :, :])
            x = x.view(x.size(0), -1)
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
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
        x_rnn = self.bn2(self.fc2(x_rnn))
        x_rnn = F.relu(x_rnn)
        x_rnn = self.dropout_layer1(x_rnn)
        x_rnn = self.fc3(x_rnn)
        return x_rnn.view(X_3d.size(0), -1)


""" LSTM + 2 imgs"""


class KinematicLSTMv1(nn.Module):
    def __init__(self, lstm_hidden_dim=1000, n_lstm_hidden_layers=1, drop_p=0.5,
                 h_fc_dim=1000, n_output=8):
        super(KinematicLSTMv1, self).__init__()

        self.lstm_input_dim = 1000
        self.lstm_hidden_dim = lstm_hidden_dim
        self.n_lstm_hidden_layers = n_lstm_hidden_layers
        self.drop_p = drop_p
        self.h_fc_dim = h_fc_dim
        self.n_output = n_output

        self.resnet = models.resnet18()

        self.label_mlp1 = nn.Linear(self.lstm_input_dim + 8, self.lstm_hidden_dim)

        self.LSTM = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.n_lstm_hidden_layers,
            batch_first=True,
        )

        self.fc1 = nn.Linear(3 * self.lstm_hidden_dim, self.h_fc_dim)
        self.fc2 = nn.Linear(self.h_fc_dim, 256)
        self.fc3 = nn.Linear(256, self.n_output)

    def forward(self, X_3d, Y_in):
        # X shape: Batch x Sequence x 3 Channels x img_dims
        # Y_in: Batch x Sequence x 8
        # Run resnet sequentially on the data to generate embedding sequence
        cnn_embed_seq = []
        for t in range(X_3d.size(1)):
            X = self.resnet(X_3d[:, t, :, :, :])
            X = F.dropout(X, p=self.drop_p, training=self.training)
            cnn_embed_seq.append(X)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

        # Pass all elements of cnn_embed_seq except last 2 through lstm
        context_embeds = cnn_embed_seq[:, :-2, :]
        query_embeds = cnn_embed_seq[:, -2:, :]
        query_embeds = query_embeds.contiguous().view(query_embeds.size(0), -1)

        # Combine context labels and context_embeds
        context_embeds = torch.cat((context_embeds, Y_in), dim=-1)
        context_embeds = self.label_mlp1(context_embeds)
        context_embeds = F.relu(context_embeds)

        # run lstm on the embedding sequence
        self.LSTM.flatten_parameters()

        RNN_out, (h_n, h_c) = self.LSTM(context_embeds, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x_rnn = torch.cat((RNN_out[:, -1, :], query_embeds), dim=1)
        # x_rnn = torch.cat((RNN_out, query_embeds), dim=1).view(RNN_out.size(0), -1)
        x_rnn = self.fc1(x_rnn)
        x_rnn = F.relu(x_rnn)
        # x_rnn = F.dropout(x_rnn, p=self.drop_p, training=self.training)
        x_rnn = self.fc2(x_rnn)
        x_rnn = F.relu(x_rnn)
        # x_rnn = F.dropout(x_rnn, p=self.drop_p, training=self.training)
        x_rnn = self.fc3(x_rnn)
        return x_rnn


class RigidTransformV0(nn.Module):
    def __init__(self, drop_p=0.5, n_output=8):
        super(RigidTransformV0, self).__init__()

        self.lstm_input_dim = 1000
        self.drop_p = drop_p
        self.n_output = n_output

        self.resnet = models.resnet18()

        self.fc1 = nn.Linear(2000, 1000)
        self.fc2 = nn.Linear(1000, 512)
        self.fc3 = nn.Linear(512, self.n_output)

    def forward(self, X_3d):
        # X shape: Batch x 2images x 3 Channels x img_dims
        # Run resnet sequentially on the data to generate embedding sequence
        cnn_embed_seq = []
        for t in range(X_3d.size(1)):
            X = self.resnet(X_3d[:, t, :, :, :])
            X = F.dropout(X, p=self.drop_p, training=self.training)
            cnn_embed_seq.append(X)

        # swap time and sample dim such that result has (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

        # stacking time dim in a single tensor
        cnn_embed_seq = cnn_embed_seq.contiguous().view(cnn_embed_seq.size(0), -1)

        # FC layers
        x_rnn = self.fc1(cnn_embed_seq)
        x_rnn = F.relu(x_rnn)
        x_rnn = self.fc2(x_rnn)
        x_rnn = F.relu(x_rnn)
        x_rnn = self.fc3(x_rnn)
        return x_rnn


def articulation_lstm_loss_spatial_distance(pred, target, wt_on_ortho=1., wt_on_ax_std=0.0):
    """ Based on Spatial distance"""
    pred = pred.view(pred.size(0), -1, 8)[:, 1:, :]  # We don't need the first row as it is for single image

    # Spatial Distance loss
    dist_err = orientation_difference_bw_plucker_lines(target, pred) ** 2 + distance_bw_plucker_lines(target, pred) ** 2

    # Configuration Loss
    conf_err = ((target[:, :, 6:] - pred[:, :, 6:]) ** 2).sum(dim=-1)

    err = dist_err + conf_err
    loss = torch.mean(err)

    # Ensure l_hat has norm 1.
    loss += torch.mean((torch.norm(pred[:, :, :3], dim=-1) - 1.) ** 2)

    # Ensure orthogonality between l_hat and m
    loss += wt_on_ortho * torch.mean(torch.abs(torch.sum(torch.mul(pred[:, :, :3], pred[:, :, 3:6]), dim=-1)))

    # # Penalize spread of screw axis
    # loss += wt_on_ax_std * (torch.mean(err.std(dim=1)[:6]))

    if torch.isnan(loss):
        print("target: Min: {},  Max{}".format(target.min(), target.max()))
        print("Prediction: Min: {},  Max{}".format(pred.min(), pred.max()))
        print("L2 error: {}".format(torch.mean((target - pred) ** 2)))
        print("Distance loss:{}".format(torch.mean(orientation_difference_bw_plucker_lines(target, pred) ** 2)))
        print("Orientation loss:{}".format(torch.mean(distance_bw_plucker_lines(target, pred) ** 2)))
        print("Configuration loss:{}".format(torch.mean(conf_err)))

    return loss


def articulation_lstm_loss_RT(pred, target, wt_on_ortho=0., extra_indiv_wts=None):
    """ For rigid transforms"""
    err = (pred - target) ** 2
    loss = torch.mean(err)

    # Ensure orthogonality between l_hat and m
    loss += wt_on_ortho * torch.mean(torch.abs(torch.sum(torch.mul(pred[:, :3], pred[:, 3:6]), dim=-1)))

    if extra_indiv_wts is None:
        extra_indiv_wts = [0., 0., 0.]

    # Extra weight on axis errors 'l'
    loss += torch.mean(extra_indiv_wts[0] * err[:, :3])

    # Extra weight on axis errors 'm'
    loss += torch.mean(extra_indiv_wts[1] * err[:, 3:6])

    # Extra weight on configuration errors
    loss += torch.mean(extra_indiv_wts[2] * err[:, 6:])
    return loss


def articulation_lstm_loss_L1(pred, target, wt_on_ax_std=1.0, wt_on_ortho=1., extra_indiv_wts=None):
    """ L1 loss function"""
    pred = pred.view(pred.size(0), -1, 8)[:, 1:, :]  # We don't need the first row as it is for single image

    err = torch.abs(pred - target)
    loss = torch.mean(err)

    # Penalize spread of screw axis
    loss += wt_on_ax_std * (torch.mean(err.std(dim=1)[:6]))

    # Ensure orthogonality between l_hat and m
    loss += wt_on_ortho * torch.mean(torch.abs(torch.sum(torch.mul(pred[:, :, :3], pred[:, :, 3:6]), dim=-1)))

    if extra_indiv_wts is None:
        extra_indiv_wts = [0., 0., 0.]

    # Extra weight on axis errors 'l'
    loss += torch.mean(extra_indiv_wts[0] * err[:, :, :3])

    # Extra weight on axis errors 'm'
    loss += torch.mean(extra_indiv_wts[1] * err[:, :, 3:6])

    # Extra weight on configuration errors
    loss += torch.mean(extra_indiv_wts[2] * err[:, :, 6:])
    return loss


def articulation_lstm_loss_L2(pred, target, wt_on_ax_std=1.0, wt_on_ortho=1., extra_indiv_wts=None):
    """ L2 loss"""
    pred = pred.view(pred.size(0), -1, 8)[:, 1:, :]  # We don't need the first row as it is for single image

    err = (pred - target) ** 2
    loss = torch.mean(err)

    # Penalize spread of screw axis
    loss += wt_on_ax_std * (torch.mean(err.std(dim=1)[:6]))

    # Ensure orthogonality between l_hat and m
    loss += wt_on_ortho * torch.mean(torch.abs(torch.sum(torch.mul(pred[:, :, :3], pred[:, :, 3:6]), dim=-1)))

    if extra_indiv_wts is None:
        extra_indiv_wts = [0., 0., 0.]

    # Extra weight on axis errors 'l'
    loss += torch.mean(extra_indiv_wts[0] * err[:, :, :3])

    # Extra weight on axis errors 'm'
    loss += torch.mean(extra_indiv_wts[1] * err[:, :, 3:6])

    # Extra weight on configuration errors
    loss += torch.mean(extra_indiv_wts[2] * err[:, :, 6:])
    return loss
