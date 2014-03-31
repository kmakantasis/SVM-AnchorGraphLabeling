# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import RandomizedPCA



def SVMs(trainData, testData, trainLabels, testLabels, representatives):
    repLabels = np.concatenate((np.zeros(100,), (np.ones(100,))), axis=0)
    
    trainData = np.concatenate((trainData, representatives), axis=0)
    trainLabels = np.concatenate((trainLabels, repLabels), axis=0)
    sample_weight_representatives = np.ones(len(trainData))
    sample_weight_representatives[1401:] *= 5
    
    clf_weights = svm.SVC()
    clf_weights.fit(trainData, trainLabels, sample_weight=sample_weight_representatives)

    clf_no_weights = svm.SVC()
    clf_no_weights.fit(trainData, trainLabels)

    predictions_weights = clf_weights.predict(testData)
    predictions_no_weights = clf_no_weights.predict(testData)

    pred_err_weights = [i for i in range(len(predictions_weights)) if predictions_weights[i] != testLabels[i]]
    pred_err_rat_weights = len(pred_err_weights)/float(len(predictions_weights)) 
    
    pred_err_no_weights = [i for i in range(len(predictions_no_weights)) if predictions_no_weights[i] != testLabels[i]]
    pred_err_rat_no_weights = len(pred_err_no_weights)/float(len(predictions_no_weights)) 
    
    return predictions_weights, predictions_no_weights, pred_err_rat_weights, pred_err_rat_no_weights
    
   
   
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
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(16), importances[range(16)], color="r", align="center")
    plt.xticks(range(16), [r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$', r'$x_5$',
                          r'$x_6$', r'$x_7$', r'$x_8$', r'$x_9$', r'$x_{10}$', 
                          r'$x_{11}$', r'$x_{12}$', r'$x_{13}$', r'$x_{14}$', r'$x_{15}$', 
                          r'$x_{16}$'])
    plt.yticks([0.0, 0.05, 0.10, 0.15, 0.20, 0.25], [r'$0.00$', r'$0.05$', r'$0.10$', r'$0.15$', r'$0.20$', r'$0.25$'])  
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xlim([-1, 16])
    plt.show()
    
    return importances
    


def PlotSVMs(trainData, trainLabels):
    pca = RandomizedPCA(n_components=2)
    pca.fit(trainData.T)
    tData = pca.components_
    X = tData.T
    y = trainLabels    
    
    n_sample = len(X)
    np.random.seed(0)
    order = np.random.permutation(n_sample)
    X = X[order]
    y = y[order].astype(np.float)

    X_train = X[:int(.9 * n_sample)]
    y_train = y[:int(.9 * n_sample)]
    X_test = X[int(.9 * n_sample):]
    #y_test = y[int(.9 * n_sample):]

    # fit the model
    fig = plt.figure(1)
    for fig_num, c in enumerate((1, 3, 9)):
        clf = svm.SVC(C=c)
        clf.fit(X_train, y_train)

        fig.add_subplot(1, 3, fig_num + 1)

        #plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired)

        # Circle out the test data
        plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=10)
        x_min = X[:, 0].min()
        x_max = X[:, 0].max()
        y_min = X[:, 1].min()
        y_max = X[:, 1].max()

        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(XX.shape)
        plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
        plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

        plt.title('C = %d' % c)
        
    plt.show(1)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    