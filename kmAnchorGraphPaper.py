# -*- coding: utf-8 -*-

import numpy as np
import scipy.spatial.distance as sciDist

def AnchorGraph(TrainData, Anchor, s, flag, cn):
    d, m, n = Anchor.shape[1], Anchor.shape[0], TrainData.shape[0]
    Z = np.zeros((n,m))
    
    Dis = sciDist.cdist(TrainData, Anchor, metric='euclidean')**2
    val = np.zeros((n,s))
    pos = np.copy(val)
    
    for i in range(s):
        val[:,i] = Dis.min(axis=1)
        pos[:,i] = Dis.argmin(axis=1)
        tep = (pos[:,i])*n+(np.asarray(range(n))).T
        Dis = Dis.T
        Dis = np.reshape(Dis,(-1,1))
        Dis[tep.astype(np.int)] = 1e60;
        Dis = np.reshape(Dis,(m,n))
        Dis = Dis.T

    ind = (pos)*(n)+np.transpose(np.tile((np.asarray(range(n))),(s,1)))
    
    if flag == 0:
        sigma = np.mean(val[:,-1]**(0.5))
        val = np.exp(-val/(1/1*sigma**2))          
        val = np.multiply(np.tile(np.sum(val, axis=1)**(-1), (s,1)).T, val)

    Z = np.reshape(Z, (-1,1))
    ind_r = np.reshape(ind.astype(np.int), (-1,1))
    val_r = np.reshape(val, (-1,1))
    for i in range(len(ind_r)):
        Z[ind_r[i]] = val_r[i]
        
    Z = np.reshape(Z, (m,n))
    Z = Z.T
     
    T = np.dot(Z.T, Z)
    rL =T - np.dot(np.dot(T, np.diag(Z.sum(axis=0)**(-1))), T)   
    
    return Z, rL
    

        
    
        
        
        
        
        
        
        
        
        
        
        
        