# -*- coding: utf-8 -*-

import numpy as np
import pylab as pl
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier


def SVMs(trainData, testData, trainLabels, testLabels):
    clf = svm.SVC()
    clf.fit(trainData, trainLabels)
    predictions = clf.predict(testData)

    pred_err = [i for i in range(len(predictions)) if predictions[i] != testLabels[i]]
    pred_err_rat = len(pred_err)/float(len(predictions)) 
    
    return predictions, pred_err_rat
    
def FeaturesImportance(trainData, trainLabels):
    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
    forest.fit(trainData, trainLabels)
    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(16):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    pl.figure()
    pl.title("Feature importances")
    pl.bar(range(16), importances[range(16)], color="r", align="center")
    pl.xticks(range(16), [r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$', r'$x_5$',
                          r'$x_6$', r'$x_7$', r'$x_8$', r'$x_9$', r'$x_{10}$', 
                          r'$x_{11}$', r'$x_{12}$', r'$x_{13}$', r'$x_{14}$', r'$x_{15}$', 
                          r'$x_{16}$'])
    pl.yticks([0.0, 0.05, 0.10, 0.15, 0.20, 0.25], [r'$0.00$', r'$0.05$', r'$0.10$', r'$0.15$', r'$0.20$', r'$0.25$'])  
    pl.xlabel('Features')
    pl.ylabel('Importance')
    pl.xlim([-1, 16])
    pl.show()
    
    return importances