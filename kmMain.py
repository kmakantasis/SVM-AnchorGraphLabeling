# -*- coding: utf-8 -*-
# Main script for anchor graph

import numpy as np
import scipy.io
import kmAnchorGraphPaper


data_mat = scipy.io.loadmat('data.mat')
data = data_mat['data']
labels = data_mat['labels']
rep_0 = data_mat['rep_0']
rep_1 = data_mat['rep_1']

representatives = np.concatenate((rep_0, rep_1))
label_index = np.zeros((1,len(representatives)))
for j in range(len(representatives)):      
    for i in range(len(data)):
        if np.array_equal(data[i], representatives[j]):
            label_index[0,j] = i
            

label_index = label_index.astype(np.int)

Z, rL = kmAnchorGraphPaper.AnchorGraph(data, representatives, 10, 0, 15)

