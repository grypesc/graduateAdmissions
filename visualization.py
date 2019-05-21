import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

data = pd.read_csv('datasetVer1.1.csv')
xData = data[['Serial No.','GRE Score','TOEFL Score','University Rating','SOP','LOR' ,'CGPA','Research']]
yData = data['Chance of Admit']

fig,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(data.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
plt.show()

print("No research:",len(data[data.Research == 0]))
print("Did a research:",len(data[data.Research == 1]))
y = np.array([len(data[data.Research == 0]),len(data[data.Research == 1])])
x = ["No research","Did a research"]
plt.bar(x,y)
plt.title("Research Experience")
plt.xlabel("Canditates")
plt.ylabel("Number of candidates")
plt.show()

data["GRE Score"].plot(kind = 'hist',bins = 50,figsize = (10,6))
plt.title("GRE Scores")
plt.xlabel("GRE Score")
plt.ylabel("Number of candidates")
plt.show()

y = np.array([data["TOEFL Score"].min(),data["TOEFL Score"].mean(),data["TOEFL Score"].max()])
x = ["Worst","Average","Best"]
plt.bar(x,y)
plt.title("TOEFL Scores")
plt.xlabel("Level")
plt.ylabel("TOEFL Score")
plt.show()

from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.1, random_state=0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(xTrain, yTrain)
print('Accuracy of SVM classifier on training set: {:.2f}'.format(reg.score(xTrain, yTrain)))
print('Accuracy of SVM classifier on test set: {:.2f}'.format(reg.score(xTest, yTest)))
