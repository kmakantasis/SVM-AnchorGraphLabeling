# -*- coding: utf-8 -*-

import numpy as np
import scipy.io
import kmAnchorGraphPaper


data_mat = scipy.io.loadmat('data.mat')
data = data_mat['data']
labels = data_mat['labels']
rep_0 = data_mat['rep_0']
rep_1 = data_mat['rep_1']

Z, rL = kmAnchorGraphPaper.AnchorGraph(data, np.concatenate((rep_0, rep_1), axis=0), 10, 0, 15)

