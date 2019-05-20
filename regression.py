import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge, Lasso, SGDRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor

kFoldSplits = 5;
data = pd.read_csv('datasetVer1.1.csv')
xLabels = ['GRE Score','TOEFL Score','University Rating','SOP','LOR' ,'CGPA','Research'];
yLabels = ['Chance of Admit'];
regressors = [
    LinearRegression(),
    Ridge(alpha=0.5),
    RandomForestRegressor(),
    DecisionTreeRegressor(random_state = 1),
    BayesianRidge(),
    LinearSVR(epsilon = 0.001),
    Lasso(alpha=0.0001),
    SGDRegressor(loss="squared_loss", penalty=None, alpha = 0.1)
    ]

from sklearn.preprocessing import MinMaxScaler
scalerX = MinMaxScaler(feature_range=(0, 1))
data[data.columns] = scalerX.fit_transform(data[data.columns])

from sklearn.model_selection import KFold
kfold = KFold(n_splits=kFoldSplits, random_state=1, shuffle=True)

for regressor in regressors:
    score = 0
    for trainDataKFoldIndex, testDataKFoldIndex in kfold.split(data):
        trainData = data.drop(testDataKFoldIndex)
        testData = data.drop(trainDataKFoldIndex)
        xTrain = trainData[xLabels]
        yTrain = trainData[yLabels]
        xTest = testData[xLabels]
        yTest = testData[yLabels]
        reg = regressor.fit(xTrain, yTrain)
        score+=reg.score(xTest, yTest)
    print("### " + regressor.__class__.__name__ + " ###")
    score = score/kFoldSplits
    print("Average accuracy on training sets: " + str(score) +"\n")
