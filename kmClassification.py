# -*- coding: utf-8 -*-

from sklearn import svm

def SVMs(trainData, testData, trainLabels, testLabels):
    clf = svm.SVC()
    clf.fit(trainData, trainLabels)
    predictions = clf.predict(testData)

    pred_err = [i for i in range(len(predictions)) if predictions[i] != testLabels[i]]
    pred_err_rat = len(pred_err)/float(len(predictions)) 
    
    return predictions, pred_err_rat