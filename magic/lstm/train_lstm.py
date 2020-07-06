import argparse

import torch
import numpy as np
from ArticulationModelLearning.magic.lstm.dataset import ArticulationDataset, RigidTransformDataset, \
    ArticulationDatasetV1
from ArticulationModelLearning.magic.lstm.model_trainer import ModelTrainer
from ArticulationModelLearning.magic.lstm.models import RigidTransformV0, KinematicLSTMv1, \
	articulation_lstm_loss_RT, articulation_lstm_loss_spatial_distance, \
    DeepArtModel, articulation_lstm_loss_L2
from ArticulationModelLearning.magic.lstm.models_v1 import DeepArtModel_v1, articulation_lstm_loss_spatial_distance_v1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train object learner on articulated object dataset.")
    parser.add_argument('--name', type=str, help='jobname', default='test')
    parser.add_argument('--train-dir', type=str, default='../data/test/microwave/')
    parser.add_argument('--test-dir', type=str, default='../data/test/microwave/')
    parser.add_argument('--ntrain', type=int, default=1,
                        help='number of total training samples (n_object_instants)')
    parser.add_argument('--ntest', type=int, default=1, help='number of test samples (n_object_instants)')
    parser.add_argument('--aug-multi', type=int, default=120, help='Multiplier for data augmentation')
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
    parser.add_argument('--model-type', type=str, default='lstm', help='lstm, rt, lstm_rt, lst_aug')
    parser.add_argument('--load-wts', action='store_true', default=False, help='Should load model wts from prior run?')
    parser.add_argument('--wts-dir', type=str, default='models/', help='Dir of saved model wts')
    parser.add_argument('--prior-wts', type=str, default='test', help='Name of saved model wts')
    parser.add_argument('--fix-seed', action='store_true', default=False, help='Should fix seed or not')

    args = parser.parse_args()

    print(args)
    print('cuda?', torch.cuda.is_available())

    ntrain = args.ntrain * args.aug_multi
    ntest = args.ntest * args.aug_multi

    if args.fix_seed:
        torch.manual_seed(1)
        np.random.seed(1)


    # elif args.model_type == 'lstm_aug':
    #     trainset = ArticulationDatasetV2(ntrain,
    #                                      args.train_dir,
    #                                      n_dof=args.ndof)
    #
    #     testset = ArticulationDatasetV2(ntest,
    #                                     args.test_dir,
    #                                     n_dof=args.ndof)
    #
    #     # init model
    #     network = KinematicLSTMv0(lstm_hidden_dim=1000, n_lstm_hidden_layers=1,
    #                               drop_p=args.drop_p, h_fc_dim=256, n_output=120)

    if args.model_type == 'rt':
        '''Rigid Transform Datasets'''
        trainset = RigidTransformDataset(ntrain,
                                         args.train_dir,
                                         n_dof=args.ndof)

        testset = RigidTransformDataset(ntest,
                                        args.test_dir,
                                        n_dof=args.ndof)

        loss_fn = articulation_lstm_loss_RT

        network = RigidTransformV0(drop_p=args.drop_p, n_output=8)

    elif args.model_type == 'lstm_rt':
        ## Sequence and 2 images
        trainset = ArticulationDatasetV1(ntrain,
                                         args.train_dir,
                                         n_dof=args.ndof)

        testset = ArticulationDatasetV1(ntest,
                                        args.test_dir,
                                        n_dof=args.ndof)

        loss_fn = articulation_lstm_loss_RT

        network = KinematicLSTMv1(lstm_hidden_dim=1000, n_lstm_hidden_layers=1,
                                  drop_p=args.drop_p, h_fc_dim=256, n_output=8)

    else:  # Default: 'lstm'
        trainset = ArticulationDataset(ntrain,
                                       args.train_dir,
                                       n_dof=args.ndof)

        testset = ArticulationDataset(ntest,
                                      args.test_dir,
                                      n_dof=args.ndof)

        loss_fn = articulation_lstm_loss_L2
        # loss_fn = articulation_lstm_loss_spatial_distance
        # loss_fn = articulation_lstm_loss_spatial_distance_v1

        # init model
        # network = KinematicLSTMv0(lstm_hidden_dim=1000, n_lstm_hidden_layers=1,
        #                           drop_p=args.drop_p, h_fc_dim=256, n_output=8)
        # network = DeepArtModel(lstm_hidden_dim=1000, n_lstm_hidden_layers=1,
        #                        drop_p=args.drop_p, h_fc_dim=256, n_output=8)
        network = DeepArtModel_v1(lstm_hidden_dim=1000, n_lstm_hidden_layers=1,
                                  drop_p=args.drop_p, n_output=8)

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch,
                                             shuffle=True, num_workers=args.nwork,
                                             pin_memory=True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch,
                                              shuffle=True, num_workers=args.nwork,
                                              pin_memory=True)

    # Load Saved wts
    if args.load_wts:
        network.load_state_dict(torch.load(args.wts_dir + args.prior_wts + '.net'))

    # setup trainer
    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')

    optimizer = torch.optim.Adam(network.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    ## Debug
    torch.autograd.set_detect_anomaly(True)

    trainer = ModelTrainer(model=network,
                           train_loader=trainloader,
                           test_loader=testloader,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           criterion=loss_fn,
                           epochs=args.epochs,
                           name=args.name,
                           test_freq=args.test_freq,
                           device=args.device,
                           obj=args.obj,
                           ndof=args.ndof)

    # train
    best_model = trainer.train()

    # #Test best model
    # trainer.test_best_model(best_model, fname_suffix='_posttraining')
