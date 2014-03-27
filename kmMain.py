# -*- coding: utf-8 -*-
# Main script for anchor graph

import numpy as np
import scipy.io
import kmAnchorGraphPaper
import kmClassification


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
            

ground = labels + 1
label_index = label_index.astype(np.int)

Z, rL = kmAnchorGraphPaper.AnchorGraph(data, representatives, 10, 0, 15)
F, A, ss_err = kmAnchorGraphPaper.AnchorGraphReg(Z, rL, ground, label_index, 0.01)

SSLabels = np.zeros((F.shape[0],))
for i in range(len(SSLabels)):
    SSLabels[i] = np.argmax(F[i,:])
    

trainData = np.concatenate((data[0:800], data[1000:1600]))
trainLabels = np.concatenate((SSLabels[0:800], SSLabels[1000:1600]))
testData = np.concatenate((data[801:999], data[1601:-1]))
testLabels = np.concatenate((SSLabels[801:999], SSLabels[1601:-1]))


predictions, clf_err = kmClassification.SVMs(trainData, testData, trainLabels, testLabels)


