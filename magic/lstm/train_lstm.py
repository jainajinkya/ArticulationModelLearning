import argparse

import torch
from ArticulationModelLearning.magic.lstm.dataset import ArticulationDataset
from ArticulationModelLearning.magic.lstm.model_trainer import ModelTrainer
from ArticulationModelLearning.magic.lstm.models import KinematicLSTMv0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train object learner on articulated object dataset.")
    parser.add_argument('--name', type=str, help='jobname', default='test')
    parser.add_argument('--train-dir', type=str, default='../../../data/test/microwave/')
    parser.add_argument('--test-dir', type=str, default='../../../data/test/microwave/')
    parser.add_argument('--ntrain', type=int, default=1,
                        help='number of total training samples (n_object_instants)')
    parser.add_argument('--ntest', type=int, default=1, help='number of test samples (n_object_instants)')
    parser.add_argument('--epochs', type=int, default=10, help='number of iterations through data')
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    parser.add_argument('--nwork', type=int, default=8, help='num_workers')
    parser.add_argument('--test-freq', type=int, default=20, help='frequency at which to test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', action='store_true', default=False, help='use cuda')
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--ndof', type=int, default=1, help='how many degrees of freedom in the object class?')
    parser.add_argument('--obj', type=str, default='microwave')
    parser.add_argument('--drop_p', type=float, default=0.5, help='dropout prob')
    parser.add_argument('--device', type=int, default=0, help='cuda device')
    args = parser.parse_args()

    print(args)
    print('cuda?', torch.cuda.is_available())

    trainset = ArticulationDataset(args.ntrain,
                                   args.train_dir,
                                   n_dof=args.ndof)

    testset = ArticulationDataset(args.ntest,
                                  args.test_dir,
                                  n_dof=args.ndof)

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch,
                                             shuffle=True, num_workers=args.nwork,
                                             pin_memory=True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch,
                                              shuffle=True, num_workers=args.nwork,
                                              pin_memory=True)

    # init model
    network = KinematicLSTMv0(lstm_hidden_dim=1000, n_lstm_hidden_layers=1,
                              drop_p=args.drop_p, h_fc_dim=256)

    # setup trainer
    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')

    optimizer = torch.optim.Adam(network.parameters(),
                                 lr=args.learning_rate)

    trainer = ModelTrainer(model=network,
                           train_loader=trainloader,
                           test_loader=testloader,
                           optimizer=optimizer,
                           epochs=args.epochs,
                           name=args.name,
                           test_freq=args.test_freq,
                           device=args.device,
                           obj=args.obj,
                           ndof=args.ndof)

    # Testing if model is learning anything
    trainer.test_best_model(trainer.model, fname_suffix='_pretraining')

    # train
    best_model = trainer.train()

    # Test best model
    trainer.test_best_model(best_model, fname_suffix='_posttraining')

