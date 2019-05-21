from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

wh = pd.read_csv('datasetVer1.1.csv')
clusters = 2
properKmeans = 0
properGausian = 0
properHierarchical = 0

data = wh[['GRE Score','TOEFL Score','University Rating','SOP','LOR' ,'CGPA', 'Chance of Admit']]
data2 = wh[["Research"]]
cor = data.corr() #Calculate the correlation of the above variables
sns.heatmap(cor, square=True) #Plot the correlation as heat map

#Scaling of data
ss = StandardScaler()
ss.fit_transform(data)

#mapping features into one. Now we can use doKmeans()
#with x_pca instead of data
pca = PCA(n_components = 1, whiten= True )  # whitten = normalize
pca.fit(data)
x_pca = pca.transform(data)

#clustering by different methods
hierarchical_cluster = AgglomerativeClustering(n_clusters = clusters,affinity= "euclidean",linkage = "ward")
clusters_hierarchical = hierarchical_cluster.fit_predict(data)
data["label_hierarchical"] = clusters_hierarchical

kmeans = KMeans(n_clusters=clusters)
clusters_knn = kmeans.fit_predict(data)
data["label_kmeans"] = clusters_knn

gausianMixture = GaussianMixture(n_components=clusters,init_params='kmeans')
clusters_gausianMixture = gausianMixture.fit_predict(data)
data["label_gausian"] = clusters_gausianMixture

def plotScatter(label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if label == "Research":
        scatter = ax.scatter(data['TOEFL Score'],data['Chance of Admit'],c=data2["Research"],s=50)
        ax.set_title('Research')
    else:
        scatter = ax.scatter(data['TOEFL Score'],data['Chance of Admit'],c=data[label],s=50)
        label=label[6:]
        ax.set_title(label+' clustering')
    ax.set_xlabel('TOEFL Score')
    ax.set_ylabel('Chance of Admit')
    plt.colorbar(scatter)
    plt.show()

#clustering gives values of 0 and 1 randomly to the Research.
#In case in which properKmeans is too small it means clustering
#asigned 0 instead of 1 to predicted researches
def checkError(labelProperCounts, name):
    if (labelProperCounts > len(output)/2):
        error = labelProperCounts/len(output)
    if (labelProperCounts < len(output)/2):
        error = (len(output)-labelProperCounts)/len(output)
    print("\nAverage error for "+name+": %.6f"%error)

#Plot real results
plotScatter("Research")
plotScatter("label_kmeans")
plotScatter("label_gausian")
plotScatter("label_hierarchical")

#writing to output file
classification = pd.DataFrame(data, columns=['Serial No.','GRE Score','TOEFL Score','University Rating','SOP','LOR'
,'CGPA', 'Chance of Admit', 'label_kmeans', "label_hierarchical", "label_gausian" ]).round(3).to_csv('./clusteringOutputs/kmeans.csv')

#calculating error
output = pd.read_csv('./clusteringOutputs/kmeans.csv')
researches = wh[['Research']]
label_kmeans = output[["label_kmeans"]]
label_gausian = output[["label_gausian"]]
label_hierarchical = output[["label_hierarchical"]]

researches = np.squeeze(np.asarray(researches))
label_kmeans = np.squeeze(np.asarray(label_kmeans))
label_gausian = np.squeeze(np.asarray(label_gausian))
label_hierarchical = np.squeeze(np.asarray(label_hierarchical))

for i in range(len(output)):
    if researches[i] != label_kmeans[i]:
        properKmeans+=1
    if researches[i] != label_gausian[i]:
        properGausian+=1
    if researches[i] != label_hierarchical[i]:
        properHierarchical+=1

print('\n*** Errors ***')
checkError(properKmeans, "K means")
checkError(properGausian, "Gausian Mixture")
checkError(properHierarchical, "Hierarchical Clustering")
print('\n\n')
