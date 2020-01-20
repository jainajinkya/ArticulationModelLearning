import os
import sys
import time

import numpy as np
import torch
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

from ArticulationModelLearning.magic.lstm.models import articulation_lstm_loss
from ArticulationModelLearning.magic.lstm.utils import interpret_label


class ModelTrainer(object):
    def __init__(self,
                 model,
                 train_loader,
                 test_loader,
                 optimizer,
                 epochs,
                 name,
                 test_freq,
                 device,
                 obj='microwave',
                 ndof=1):
        super(ModelTrainer, self).__init__()
        self.model = model
        self.trainloader = train_loader
        self.testloader = test_loader
        self.optimizer = optimizer
        self.criterion = articulation_lstm_loss
        self.epochs = epochs
        self.name = name
        self.test_freq = test_freq
        self.obj = obj
        self.ndof = ndof

        self.losses = []
        self.tlosses = []

        # float model as push to GPU/CPU
        self.device = device
        self.model.float().to(self.device)

    def train(self):
        best_tloss = 1e8
        for epoch in range(self.epochs):
            sys.stdout.flush()
            loss = self.train_epoch(epoch)
            self.losses.append(loss)
            if epoch % self.test_freq == 0:
                tloss = self.test_epoch(epoch)
                self.tlosses.append(tloss)
                self.plot_losses()

                if tloss < best_tloss:
                    print('saving model.')
                    net_fname = 'models/' + str(self.name) + '.net'
                    torch.save(self.model.state_dict(), net_fname)
                    best_tloss = tloss

        # plot losses one more time
        self.plot_losses()

        # re-load the best state dictionary that was saved earlier.
        self.model.load_state_dict(torch.load(net_fname, map_location='cpu'))

        return self.model

    def train_epoch(self, epoch):
        start = time.time()
        running_loss = 0
        batches_per_dataset = len(self.trainloader.dataset) / self.trainloader.batch_size
        for i, X in enumerate(self.trainloader):
            self.optimizer.zero_grad()
            depth, labels = X['depth'].to(self.device), X['label'].to(self.device)

            y_pred = self.model(depth)
            loss = self.criterion(y_pred, labels)
            if loss.data == -float('inf'):
                print('inf loss caught, not backpropping')
                running_loss += -1000
            else:
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

        stop = time.time()
        print('Epoch %s -  Train  Loss: %.5f Time: %.5f' % (str(epoch).zfill(3),
                                                            running_loss / batches_per_dataset,
                                                            stop - start))
        return running_loss / batches_per_dataset

    def test_epoch(self, epoch):
        start = time.time()
        running_loss = 0
        batches_per_dataset = len(self.testloader.dataset) / self.testloader.batch_size
        with torch.no_grad():
            for i, X in enumerate(self.testloader):
                depth, labels = X['depth'].to(self.device), X['label'].to(self.device)
                y_pred = self.model(depth)
                loss = self.criterion(y_pred, labels)
                running_loss += loss.item()

        stop = time.time()
        print('Epoch %s -  Test  Loss: %.5f Euc. Time: %.5f' % (str(epoch).zfill(3),
                                                                running_loss / batches_per_dataset,
                                                                stop - start))
        return running_loss / batches_per_dataset

    def plot_losses(self):
        os.makedirs("plots/" + self.name, exist_ok=True)
        x = np.arange(len(self.losses))
        tx = np.arange(0, len(self.losses), self.test_freq)
        plt.plot(x, np.array(self.losses), color='b', label='train')
        plt.plot(tx, np.array(self.tlosses), color='r', label='test')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('plots/' + self.name + '/curve.png')
        plt.close()
        np.save('plots/' + self.name + '/losses.npy', np.array(self.losses))
        np.save('plots/' + self.name + '/tlosses.npy', np.array(self.tlosses))

    def test_best_model(self, best_model):
        all_l_hat_stds = []
        all_m_stds = []
        with torch.no_grad():
            for i, X in enumerate(self.testloader):
                depth, labels = X['depth'].to(self.device), X['label'].to(self.device)
                y_pred = best_model(depth)

                for j in range(y_pred.size(0)):
                    pred_label = interpret_label(y_pred[j, :])
                    all_l_hat_stds.append(pred_label['screw_axis_std'][0].numpy())
                    all_m_stds.append(pred_label['screw_axis_std'][1].numpy())

        # Plot variation of screw axis
        nbins = 32
        all_l_hat_stds = np.array(all_l_hat_stds)
        all_m_stds = np.array(all_m_stds)

        # l_hat
        fig1, axs1 = plt.subplots(1, 3, sharey=True)
        axs1[0].hist(all_l_hat_stds[:, 0], bins=nbins)
        axs1[1].hist(all_l_hat_stds[:, 1], bins=nbins)
        axs1[2].hist(all_l_hat_stds[:, 2], bins=nbins)

        axs1[0].set_title('x_axis')
        axs1[1].set_title('y_axis')
        axs1[2].set_title('z_axis')
        fig1.suptitle('Histogram of std in predicting l_hat', fontsize=16)
        fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('plots/' + self.name + '/l_hat_std.png')
        plt.close()

        # m
        fig2, axs2 = plt.subplots(1, 3, sharey=True)
        axs2[0].hist(all_m_stds[:, 0], bins=nbins, label='x_axis')
        axs2[1].hist(all_m_stds[:, 1], bins=nbins, label='y_axis')
        axs2[2].hist(all_m_stds[:, 2], bins=nbins, label='z_axis')

        axs2[0].set_title('x_axis')
        axs2[1].set_title('y_axis')
        axs2[2].set_title('z_axis')
        fig2.suptitle('Histogram of std in predicting m', fontsize=16)
        fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('plots/' + self.name + '/m_std.png')
        plt.close()
