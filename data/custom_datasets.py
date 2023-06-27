#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from torch.utils.data import DataLoader, Dataset
import torch
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.datasets import load_svmlight_file
import numpy as np

class Clusters(Dataset):
    def __init__(self, size=100, num_clusters=2, dim=2, sigma=0.2, uniform_range=[-1,1], cluster_centers=[]):
        assert(size%num_clusters==0)
        samples_per_cluster = int(size/num_clusters)
        if not len(cluster_centers):
            cluster_centers = np.random.uniform(low=uniform_range[0], high=uniform_range[1], size=(num_clusters, dim))
        data = np.zeros((size,dim))
        targets = np.ones(size)
        for cluster in range(num_clusters):
            norm_rand = np.random.normal(0, sigma, size=(samples_per_cluster,dim))
            data[cluster*samples_per_cluster:cluster*samples_per_cluster + samples_per_cluster,:] = np.repeat(cluster_centers[cluster,:].reshape(1,dim), samples_per_cluster, axis=0) + norm_rand
            targets[cluster*samples_per_cluster:cluster*samples_per_cluster + samples_per_cluster] = cluster
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.int64)
        self.num_clusters = num_clusters
        self.samples_per_cluster = samples_per_cluster
        self.size = size
        self.cluster_centers = cluster_centers

    def plot(self, show=True):
        print("Clusters plot limited to 2 dimensions")
        markers = [".", "v", "s", "*", "+"]
        for cluster in range(self.num_clusters):
            marker = markers[cluster]
            plt.scatter(self.data[cluster*self.samples_per_cluster:cluster*self.samples_per_cluster + self.samples_per_cluster,0],self.data[cluster*self.samples_per_cluster:cluster*self.samples_per_cluster + self.samples_per_cluster,1], marker=marker,label="cluster {}".format(cluster+1))
        if show:
            plt.show()

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return (self.data[idx], self.targets[idx])
    
class Line(Dataset):
    def __init__(self, size=100, biases=[0], sigma=0.7, uniform_range=[0,1]):
        assert(size%len(biases)==0)
        self.size = size
        size_per_bias = int(size/len(biases))
        targets = np.random.normal(biases[0]*np.ones(size_per_bias), sigma)
        for bias in biases[1:]:
            rand = np.random.normal(bias*np.ones(size_per_bias), sigma)
            targets = np.concatenate((targets, rand))
        data = np.random.uniform(uniform_range[0], uniform_range[1], size)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
        self.data = torch.tensor(data, dtype=torch.float32).squeeze(0).unsqueeze(-1)
    
    def plot(self, show=True):
        plt.scatter(self.data, self.targets)
        if show:
            plt.show()

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return (self.data[idx], self.targets[idx])

class EpsilonNormalized(Dataset):
    def __init__(self, train=True, ratio=0.6, sort=False):
        with open("data\epsilon\epsilon_t.pickle", 'rb') as f:
            self.data, self.targets = pickle.load(f)
            if train:
                self.data = self.data[0:int(self.targets.shape[0]*ratio),:]
                self.targets = self.targets[0:int(self.targets.shape[0]*ratio)]
            else:
                self.data = self.data[int(self.targets.shape[0]*ratio):-1,:]
                self.targets = self.targets[int(self.targets.shape[0]*ratio):-1]

        if sort:
            indices = np.argsort(self.targets)
            self.targets = self.targets[indices]
            self.data = self.data[indices,:]
        self.targets[self.targets==-1] = 0        

        self.targets = torch.tensor(self.targets, dtype=torch.int8)
        self.data = torch.tensor(self.data, dtype=torch.float32)

    def extract_and_pickle(dataset_path):
        dataset_path = os.path.expanduser('data/epsilon/epsilon_normalized.t.bz2')
        A, y = load_svmlight_file(dataset_path)
        A = A.toarray()
        with open('data\epsilon\epsilon_t.pickle', 'wb') as pickle_file:
            pickle.dump((A, y), pickle_file, protocol=4)

    def __len__(self):
        return self.targets.shape[0]
    
    def __getitem__(self, idx):
        sample = self.data[idx, :]
        label = self.targets[idx].item()
        return (sample, label)
