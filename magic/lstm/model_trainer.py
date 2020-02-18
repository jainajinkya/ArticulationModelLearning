import os
import sys
import time

import numpy as np
import torch
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

from ArticulationModelLearning.magic.lstm.utils import dual_quaternion_to_screw_batch_mode


class ModelTrainer(object):
    def __init__(self,
                 model,
                 train_loader,
                 test_loader,
                 optimizer,
                 criterion,
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
        self.criterion = criterion
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
        for epoch in range(self.epochs + 1):
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
            depth, all_labels, labels = X['depth'].to(self.device), \
                                        X['all_labels'].to(self.device), \
                                        X['label'].to(self.device)

            y_pred = self.model(depth, all_labels)
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
                depth, all_labels, labels = X['depth'].to(self.device),\
                                            X['all_labels'].to(self.device), \
                                            X['label'].to(self.device)
                y_pred = self.model(depth, all_labels)
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

    def test_best_model(self, best_model, fname_suffix='', dual_quat_mode=False):
        best_model.eval()  # Put model in evaluation mode

        all_l_hat_err = torch.empty(0)
        all_m_err = torch.empty(0)
        all_q_err = torch.empty(0)
        all_d_err = torch.empty(0)
        all_l_hat_std = torch.empty(0)
        all_m_std = torch.empty(0)
        all_q_std = torch.empty(0)
        all_d_std = torch.empty(0)

        with torch.no_grad():
            for X in self.testloader:
                depth, labels = X['depth'].to(self.device), X['label'].to(self.device)
                y_pred = best_model(depth)
                y_pred = y_pred.view(y_pred.size(0), -1, 8)

                if dual_quat_mode:
                    y_pred = dual_quaternion_to_screw_batch_mode(y_pred)
                    labels = dual_quaternion_to_screw_batch_mode(labels)

                err = labels - y_pred
                all_l_hat_err = torch.cat(
                    (all_l_hat_err, torch.mean(torch.norm(err[:, :, :3], dim=-1), dim=-1).cpu()))
                all_m_err = torch.cat((all_m_err, torch.mean(torch.norm(err[:, :, 3:6], dim=-1), dim=-1).cpu()))
                all_q_err = torch.cat((all_q_err, torch.mean(err[:, :, 6], dim=-1).cpu()))
                all_d_err = torch.cat((all_d_err, torch.mean(err[:, :, 7], dim=-1).cpu()))

                all_l_hat_std = torch.cat(
                    (all_l_hat_std, torch.std(torch.norm(err[:, :, :3], dim=-1), dim=-1).cpu()))
                all_m_std = torch.cat((all_m_std, torch.std(torch.norm(err[:, :, 3:6], dim=-1), dim=-1).cpu()))
                all_q_std = torch.cat((all_q_std, torch.std(err[:, :, 6], dim=-1).cpu()))
                all_d_std = torch.cat((all_d_std, torch.std(err[:, :, 7], dim=-1).cpu()))

        # Plot variation of screw axis
        x_axis = np.arange(all_l_hat_err.size(0))

        fig = plt.figure(1)
        plt.errorbar(x_axis, all_l_hat_err.numpy(), all_l_hat_std.numpy(), capsize=3., capthick=1.)
        plt.xlabel("Test object number")
        plt.ylabel("Error")
        plt.title("Test error in l_hat")
        plt.tight_layout()
        plt.savefig('plots/' + self.name + '/l_hat_err' + fname_suffix + '.png')
        plt.close(fig)

        fig = plt.figure(2)
        plt.errorbar(x_axis, all_m_err.numpy(), all_m_std.numpy(), capsize=3., capthick=1.)
        plt.xlabel("Test object number")
        plt.ylabel("Error")
        plt.title("Test error in m")
        plt.tight_layout()
        plt.savefig('plots/' + self.name + '/m_err' + fname_suffix + '.png')
        plt.close(fig)

        fig = plt.figure(3)
        plt.errorbar(x_axis, all_q_err.numpy(), all_q_std.numpy(), capsize=3., capthick=1.)
        plt.xlabel("Test object number")
        plt.ylabel("Error")
        plt.title("Test error in theta")
        plt.tight_layout()
        plt.savefig('plots/' + self.name + '/theta_err' + fname_suffix + '.png')
        plt.close(fig)

        fig = plt.figure(4)
        plt.errorbar(x_axis, all_d_err.numpy(), all_d_std.numpy(), capsize=3., capthick=1.)
        plt.xlabel("Test object number")
        plt.ylabel("Error")
        plt.title("Test error in d")
        plt.tight_layout()
        plt.savefig('plots/' + self.name + '/d_err' + fname_suffix + '.png')
        plt.close(fig)
