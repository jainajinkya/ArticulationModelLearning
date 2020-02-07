import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class KinematicLSTMv0(nn.Module):
    def __init__(self, lstm_hidden_dim=1000, n_lstm_hidden_layers=1, drop_p=0.5,
                 h_fc_dim=256, n_output=128):
        super(KinematicLSTMv0, self).__init__()

        self.lstm_input_dim = 1000
        self.lstm_hidden_dim = lstm_hidden_dim
        self.n_lstm_hidden_layers = n_lstm_hidden_layers
        self.drop_p = drop_p
        self.h_fc_dim = h_fc_dim
        self.n_output = n_output

        self.resnet = models.resnet18()

        self.LSTM = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.n_lstm_hidden_layers,
            batch_first=True,
        )

        self.fc1 = nn.Linear(self.lstm_hidden_dim, self.h_fc_dim)
        self.fc2 = nn.Linear(self.h_fc_dim, self.n_output)

    def forward(self, X_3d):
        # X shape: Batch x Sequence x 3 Channels x img_dims
        # Run resnet sequentially on the data to generate embedding sequence
        cnn_embed_seq = []
        for t in range(X_3d.size(1)):
            X = self.resnet(X_3d[:, t, :, :, :])
            X = F.dropout(X, p=self.drop_p, training=self.training)
            cnn_embed_seq.append(X)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

        # run lstm on the embedding sequence
        self.LSTM.flatten_parameters()

        RNN_out, (h_n, h_c) = self.LSTM(cnn_embed_seq, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x_rnn = self.fc1(RNN_out[:, -1, :])  # choose RNN_out at the last time step
        # x_rnn = F.relu(x_rnn)
        x_rnn = F.dropout(x_rnn, p=self.drop_p, training=self.training)
        x_rnn = self.fc2(x_rnn)

        return x_rnn


class RigidTransformV0(nn.Module):
    def __init__(self, drop_p=0.5, n_output=8):
        super(RigidTransformV0, self).__init__()

        self.lstm_input_dim = 1000
        self.drop_p = drop_p
        self.n_output = n_output

        self.resnet = models.resnet18()

        self.fc1 = nn.Linear(2000, 1000)
        self.fc2 = nn.Linear(1000, 256)
        self.fc3 = nn.Linear(256, self.n_output)

    def forward(self, X_3d):
        # X shape: Batch x 2images x 3 Channels x img_dims
        # Run resnet sequentially on the data to generate embedding sequence
        cnn_embed_seq = []
        for t in range(X_3d.size(1)):
            X = self.resnet(X_3d[:, t, :, :, :])
            X = F.dropout(X, p=self.drop_p, training=self.training)
            cnn_embed_seq.append(X)

        # import pdb; pdb.set_trace()
        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

        # stacking time dim in a single tensor
        cnn_embed_seq = cnn_embed_seq.contiguous().view(cnn_embed_seq.size(0), -1)

        # FC layers
        x_rnn = self.fc1(cnn_embed_seq)
        x_rnn = F.dropout(x_rnn, p=self.drop_p, training=self.training)
        x_rnn = self.fc2(x_rnn)
        x_rnn = F.dropout(x_rnn, p=self.drop_p, training=self.training)
        x_rnn = self.fc3(x_rnn)
        return x_rnn


def articulation_lstm_loss(pred, target, wt_on_ax_std=1.0, wt_on_ortho=1., extra_indiv_wts=None):
    pred = pred.view(pred.size(0), -1, 8)
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


def articulation_lstm_loss_RT(pred, target, wt_on_ortho=1., extra_indiv_wts=None):
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
