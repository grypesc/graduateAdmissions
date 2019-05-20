import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge, Lasso
from sklearn.svm import LinearSVR, NuSVR


kFoldSplits = 5;
data = pd.read_csv('datasetVer1.1.csv')
xLabels = ['Serial No.','GRE Score','TOEFL Score','University Rating','SOP','LOR' ,'CGPA','Research'];
yLabels = ['Chance of Admit'];
classifiers = [
    LinearRegression(),
    Ridge(alpha=0.5),
    RandomForestRegressor(),
    BayesianRidge(),
    LinearSVR(epsilon = 0),
    Lasso(),

    ]
# xData = data[['Serial No.','GRE Score','TOEFL Score','University Rating','SOP','LOR' ,'CGPA','Research']]
# yData = data['Chance of Admit']

# from sklearn.model_selection import train_test_split
# xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.1, random_state=0)

 # from sklearn.preprocessing import MinMaxScaler
 # scalerX = MinMaxScaler(feature_range=(0, 1))
 # xTrain[xTrain.columns] = scalerX.fit_transform(xTrain[xTrain.columns])
 # xTest[xTest.columns] = scalerX.transform(xTest[xTest.columns])

from sklearn.model_selection import KFold
kfold = KFold(n_splits=kFoldSplits, random_state=1, shuffle=True) # Define the split - into 2 folds
kfold.get_n_splits(data) # returns the number of splitting iterations in the cross-validator


# enumerate splits
for clf in classifiers:
    name = clf.__class__.__name__
    score = 0;
    for trainDataKFoldIndex, testDataKFoldIndex in kfold.split(data):

        trainData = data.drop(testDataKFoldIndex)
        testData = data.drop(trainDataKFoldIndex)

        xTrain = trainData[xLabels]
        yTrain = trainData[yLabels]
        xTest = testData[xLabels]
        yTest = testData[yLabels]

        reg = clf.fit(xTrain, yTrain)
        #print("*** ", name, " ***")
        #print('Accuracy on training set: {:.2f}'.format(reg.score(xTrain, yTrain)))
        #print('Accuracy on test set: {:.2f} \n\n'.format(reg.score(xTest, yTest)))
        score+=reg.score(xTest, yTest)
    print("### " + name + " ###")
    score = score/kFoldSplits
    print("Average score on training sets: " + str(score) +"\n\n")
