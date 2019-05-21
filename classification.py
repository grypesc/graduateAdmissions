import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

classifiers = [
    LogisticRegression(),
    SVC(),
    MLPClassifier(alpha=1e-4, hidden_layer_sizes=(20, 20, 20, 20)),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    DecisionTreeClassifier(),
    GaussianNB(),
    KNeighborsClassifier()
    ]

data = pd.read_csv("datasetVer1.1.csv",sep = ",")
serialNo = data["Serial No."].values
data.drop(["Serial No."],axis=1,inplace = True)

y = data["Chance of Admit"].values
x = data.drop(["Chance of Admit"],axis=1)

from sklearn.model_selection import train_test_split
xTrain, xTest,yTrain, yTest = train_test_split(x,y,test_size = 0.20,random_state = 1)

# normalization
from sklearn.preprocessing import MinMaxScaler
scalerX = MinMaxScaler(feature_range=(0, 1))
xTrain[xTrain.columns] = scalerX.fit_transform(xTrain[xTrain.columns])
xTest[xTest.columns] = scalerX.transform(xTest[xTest.columns])

yTrainClassified = [3 if each >= 0.9 else 2 if each >= 0.75 else 1 if each >= 0.50 else 0 for each in yTrain]
yTestClassified = [3 if each >= 0.9 else 2 if each >= 0.75 else 1 if each >= 0.50 else 0 for each in yTest]

yTrainClassified = np.array(yTrainClassified)
yTestClassified = np.array(yTestClassified)

from sklearn.metrics import confusion_matrix

for classifier in classifiers:
    classifier.fit(xTrain, yTrainClassified)
    print("### " + classifier.__class__.__name__ + " ###")
    print("Classification accuracy: " + str(classifier.score(xTest, yTestClassified)) +"\n")
    # print("real value of yTestClassified[1]: " + str(yTestClassified[1]) + " -> the predict: " + str(lrc.predict(xTest.iloc[[1],:])))
    # print("real value of yTestClassified[2]: " + str(yTestClassified[2]) + " -> the predict: " + str(lrc.predict(xTest.iloc[[2],:])))

    cm_lrc = confusion_matrix(yTestClassified,classifier.predict(xTest))
    f, ax = plt.subplots(figsize =(5,5))
    sns.heatmap(cm_lrc,annot = True,linewidths=0.5,linecolor="gray",fmt = ".0f",ax=ax)
    plt.title(classifier.__class__.__name__)
    plt.xlabel("Predicted classes")
    plt.ylabel("Real classes")
    plt.show()

    #from sklearn.metrics import precision_score, recall_score
    #print("precision_score: ", precision_score(yTestClassified,classifier.predict(xTest)))
    #print("recall_score: ", recall_score(yTestClassified,classifier.predict(xTest)))

    #from sklearn.metrics import f1_score
    #print("f1_score: ",f1_score(yTestClassified,classifier.predict(xTest)))
