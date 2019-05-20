import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation
from sklearn.mixture import GaussianMixture

wh = pd.read_csv('datasetVer1.1.csv')
data = wh[['GRE Score','TOEFL Score','University Rating','SOP','LOR' , 'Research', 'CGPA','Chance of Admit']]
clusteredData = wh[['GRE Score','TOEFL Score','University Rating','SOP','LOR' ,'CGPA','Chance of Admit']]

#Scaling of data
ss = StandardScaler()
ss.fit_transform(clusteredData)

# K means Clustering
def doKmeans(X, nclust=2):
    model = KMeans(nclust)
    model.fit(X)
    clust_labels = model.predict(X)
    cent = model.cluster_centers_
    return (clust_labels, cent)

clust_labels, cent = doKmeans(clusteredData, 2)


kmeans = pd.DataFrame(clust_labels)
for i in kmeans.iterrows():
    print(i)
print(kmeans)

clusteredData.insert((clusteredData.shape[1]),'kmeans',kmeans)

# Plot the clusters obtained using k means
fig = plt.figure()
ax = fig.add_subplot(111)

scatter = ax.scatter(clusteredData['TOEFL Score'],clusteredData['Chance of Admit'],c=kmeans[0],s=50)
ax.set_title('K-Means Clustering')
ax.set_xlabel('TOEFL Score')
ax.set_ylabel('Chance of Admit')
plt.colorbar(scatter)
classification = pd.DataFrame(clusteredData, columns=['GRE Score','TOEFL Score','University Rating','SOP','LOR' ,'CGPA', 'kmeans', 'Chance of Admit' ]).round(3).to_csv('kmeans.csv')
#plt.show();

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(data['TOEFL Score'],data['Chance of Admit'],c=data["Research"],s=50)
ax.set_title('Original')
ax.set_xlabel('TOEFL Score')
ax.set_ylabel('Chance of Admit')
plt.colorbar(scatter)
#plt.show();

print(data.where(data["Research"]==classification["kmeans"]))
