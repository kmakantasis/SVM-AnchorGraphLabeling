# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
import pysptools.unmixing as umx

def LoadData(filename, precomputed=True):
    if precomputed:
        data_mat = scipy.io.loadmat(filename)
        samples = data_mat['data']
        labels = data_mat['labels']
        rep_0 = data_mat['rep_0']   
        rep_1 = data_mat['rep_1']
    else:
        data_mat = scipy.io.loadmat(filename)
        features = data_mat['features']

        samples = features[:,0:16]    
        labels = features[:,-1] 
    
        samples_0 = samples[labels==0,:]
        labels_0 = labels[labels==0]
        idx = np.random.randint(low=0, high=len(samples_0), size=(1000,))
        samples_0 = samples_0[idx]
        labels_0 = labels_0[idx]
        samples_1 = samples[labels==1,:]
        labels_1 = labels[labels==1]
    
        labels = np.concatenate((labels_0, labels_1), axis=0)
        samples = np.concatenate((samples_0, samples_1), axis=0)    
    
        p = 100
        nfindr = umx.NFINDR()
        E = nfindr.unmix(samples_0[:,:,np.newaxis], p, maxit=5, normalize=True, ATGP_init=False)
        rep_0 = samples_0[np.asarray(nfindr.get_idx())[:,0],:] 

        p = 100
        nfindr = umx.NFINDR()
        E = nfindr.unmix(samples_1[:,:,np.newaxis], p, maxit=5, normalize=True, ATGP_init=False)
        rep_1 = samples_1[np.asarray(nfindr.get_idx())[:,0],:]     

        d = {}
        d['data'] = samples
        d['labels'] = labels
        d['rep_0'] = rep_0
        d['rep_1'] = rep_1

        scipy.io.savemat('data_2.mat', d)    
    
    labels = labels.T    
    
    return samples, labels, rep_0, rep_1
