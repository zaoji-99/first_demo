#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # global arguments
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='cpu or gpu')
    parser.add_argument('--seed', type=int, default='1',
                        help='random seed for random initialization')
    parser.add_argument('--n_clients', type=int, default=100,
                        help='number total clients')
    parser.add_argument('--eta', type=int, default=1,
                        help='controlling global learning rate')
    parser.add_argument('--rounds', type=int, default=2060,
                        help='communication rounds between server and edge devices')
    parser.add_argument('--record_step', type=int, default=500,
                        help='save global model every {record_step} rounds')
    parser.add_argument('--strategy', type=str, default='FedAVG',
                        help='aggregation method')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.9,
                        help='divide the training data using a Dirichlet distribution')
    parser.add_argument('--participating_ratio', type=float, default=0.1,
                        help='the fraction of total clients participating training')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='local batch size')
    parser.add_argument('--epochs', type=int, default=2,
                        help='the number of local epochs')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='SGD weight_decay')
    parser.add_argument('--resumed', type=str, default='False',
                        help='whether resume the trained model')
    parser.add_argument('--resumed_name', type=str, default='FedAVG_12192148/FedAVG_round_1000.pth',
                        help='the path of resumed model')
    parser.add_argument('--alpha', type=float, default=1,
                        help='weight of two kind of losses')
    parser.add_argument('--scale', type=float, default=0.7,
                        help='weight of two kind of losses')

    # model arguments
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset name')
    parser.add_argument('--model_name', type=str, default='cnn',
                        help='model name')

    # malicious arguments
    parser.add_argument('--trigger_type', type=str, default='pixel',
                        help='pixel pattern trigger or semantic trigger')
    parser.add_argument('--semantic_type', type=str, default='background_wall',
                        help='what kind of semantic trigger')
    parser.add_argument('--swap_class_raw', type=int, default=7,
                        help='which class we want to mis-classify')
    parser.add_argument('--batches_of_mal_data', type=int, default=10,
                        help='batches of poisoned train dataset')
    parser.add_argument('--num_mal_samples', type=int, default=15,
                        help='number of malicious samples')
    parser.add_argument('--mal_ratio', type=float, default=0.06,
                        help='fraction of malicious clients')
    parser.add_argument('--participating_mal_ratio', type=float, default=0.6,
                        help='fraction of malicious clients participating training in a given round')
    parser.add_argument('--mal_boost', type=int, default=100,
                        help='malicious boosting')
    parser.add_argument('--poison_label_swap', type=int, default=2,
                        help='flip the original label to {poison_label_swap} after injecting triggers')
    # parser.add_argument('--poison_rounds', type=str, default='0',
    #                     help='attack rounds')
    parser.add_argument('--continuity', type=str, default='False',
                        help='whether to attack continuously')
    parser.add_argument('--poison_rounds', type=int, default=60,
                        help='attack rounds')
    parser.add_argument('--poison_prob', type=float, default=0.5,
                        help='poison probability each round')
    parser.add_argument('--mal_epochs', type=int, default=6,
                        help='the number of local epochs of attackers')
    parser.add_argument('--mal_lr', type=float, default=0.05,
                        help='learning rate of attackers')
    parser.add_argument('--mal_momentum', type=float, default=0.9,
                        help='momentum of attackers')
    parser.add_argument('--mal_weight_decay', type=float, default=5e-4,
                        help='weight_decay of attackers')

    # defense
    parser.add_argument('--defense', type=str, default='no',
                        help='defense against backdoor attacks')
    parser.add_argument('--sigma', type=float, default=0.01,
                        help='std of the Laplace noise in FL-WBC')

    args = parser.parse_args()
    return args
