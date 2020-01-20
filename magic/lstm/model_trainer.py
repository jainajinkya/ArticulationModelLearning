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
        with torch.no_grad():
            for i, X in enumerate(self.testloader):
                depth, labels = X['depth'].to(self.device), X['label'].to(self.device)
                y_pred = best_model(depth)
                y_pred = y_pred.view(y_pred.size(0), -1, 8)

                err = labels - y_pred
                all_l_hat_err = torch.mean(torch.norm(err[:, :, :3], dim=-1), dim=-1).cpu().numpy()
                all_m_err = torch.mean(torch.norm(err[:, :, 3:6], dim=-1), dim=-1).cpu().numpy()
                all_q_err = torch.mean(err[:, :, 6], dim=-1).cpu().numpy()
                all_d_err = torch.mean(err[:, :, 7], dim=-1).cpu().numpy()


        # Plot variation of screw axis
        x_axis = np.arange(labels.size(0))

        # Screw Axis
        fig, axs = plt.subplots(1, 2, sharey=True)
        axs[0].plot(x_axis, all_l_hat_err)
        axs[1].plot(x_axis, all_m_err)

        axs[0].set_title('Mean error in l_hat')
        axs[1].set_title('Mean error in m')

        fig.suptitle('Screw Axis error', fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('plots/' + self.name + '/axis_err.png')
        plt.close()

        fig1, axs1 = plt.subplots(1, 2, sharey=True)
        axs1[0].plot(x_axis, all_q_err)
        axs1[1].plot(x_axis, all_d_err)

        axs1[0].set_title('Mean error in q')
        axs1[1].set_title('Mean error in d')

        fig1.suptitle('Mean configuration errors', fontsize=16)
        fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('plots/' + self.name + '/conf_errs.png')
        plt.close()
