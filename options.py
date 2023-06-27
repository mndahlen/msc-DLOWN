#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # DSGD arguments
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--local_bs', type=int, default=50, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=100, help="Size of batches for test")
    parser.add_argument('--nb', type=int, default=20, help="Number of batches for test")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--policy', type=str, default='all', help="Policy for scheduling")
    parser.add_argument('--random_policy_p', type=float, default=0.1, help="Probability to schedule device in random policy")
    parser.add_argument('--T', type=int, default=100, help="Number of learning rounds")
    parser.add_argument('--topology', type=str, default="complete", help="Graph topology")
    parser.add_argument('--comm', type=str, default="perfect", help="Communication Model")

    # machine learning arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--loss_function', type=str, default='cross_entropy', help='cross_entropy or square_error')
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    
    # other arguments
    parser.add_argument('--gpu', type=int, default=-1, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--status_freq', type=int, default=1, help='Frequency of printing learning performance')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    args = parser.parse_args()
    return args
