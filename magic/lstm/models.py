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
            # X = F.dropout(X, p=self.drop_p, training=self.training)
            cnn_embed_seq.append(X)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

        # run lstm on the embedding sequence
        self.LSTM.flatten_parameters()

        RNN_out, (h_n, h_c) = self.LSTM(cnn_embed_seq, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x_rnn = self.fc1(RNN_out[:, -1, :])   # choose RNN_out at the last time step
        # x_rnn = F.relu(x_rnn)
        # x_rnn = F.dropout(x_rnn, p=self.drop_p, training=self.training)
        x_rnn = self.fc2(x_rnn)

        return x_rnn


def articulation_lstm_loss(pred, target):
    pred = pred.view(pred.size(0), -1, 8)

    # Calculate orientation error
    loss = torch.mean((pred - target)**2)
    return loss
