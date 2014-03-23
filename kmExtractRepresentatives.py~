# -*- coding: utf-8 -*-
# Script for features extraction

import numpy as np
import scipy.io
import pysptools.unmixing as umx
from sklearn.decomposition import RandomizedPCA
import matplotlib.pyplot as plt


data_mat = scipy.io.loadmat('features.mat')
data = data_mat['features']
features = data[:,0:-1]
labels = data[:,-1]

features_label_1 = features[labels==1]
labels_label_1 = labels[labels==1]

rnd = np.random.random_integers(0, 188772, (1000,))
features_label_0 = features[labels==0]
features_label_0 = features_label_0[rnd]
labels_label_0 = labels[labels==0]
labels_label_0 = labels_label_0[rnd]

nfindr = umx.NFINDR()
num_rep = 100
E = nfindr.unmix(features_label_0[:,:,np.newaxis],  p=num_rep)
temp = np.asarray(nfindr.get_idx())
representatives_0 = temp[:,0]

E = nfindr.unmix(features_label_1[:,:,np.newaxis],  p=num_rep)
temp = np.asarray(nfindr.get_idx())
representatives_1 = temp[:,0]
representatives_1 = representatives_1 + 1000

d = {}
data = np.concatenate((features_label_0, features_label_1), axis=0)
labels = np.concatenate((labels_label_0, labels_label_1), axis=0)
rep_1 = data[representatives_1]
rep_0 = data[representatives_0]

d['data'] = data
d['labels'] = labels
d['rep_1'] = rep_1
d['rep_0'] = rep_0
scipy.io.savemat('data.mat', d)

pca = RandomizedPCA(n_components=2)
pca.fit_transform(features_label_0.T)
reduced = pca.components_.T                 

plt.plot(reduced[:,0], reduced[:,1], 'ro')
for i in range(num_rep):
    plt.plot(reduced[representatives[i],0], reduced[representatives[i],1], 'bo')


plt.show()
