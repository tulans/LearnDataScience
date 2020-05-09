import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
sns.set()
from sklearn.cluster import KMeans
data = pd.read_csv('../data/cluster-analysis/Countryclusters.csv')
print(data)

#This code could be used to plot the raw data.
#plt.scatter(data['Longitude'], data['Latitude'])
#plt.xlim(-180, 180)
#plt.ylim(-90, 90)
#plt.show()

x = data.iloc[:, 1:3]

print('Use dataFrame iLoc method to get Lat Long ')
print(x)

#Number of cluster that we want to produce is configurable here
k = 3
print('Trying '+ str(k) +' means clustering')
kmeans = KMeans(k)

#This code will apply k-means clustering with k clusters to X
kmeans.fit(x)

#To identify the clusters
identify_cluster = kmeans.fit_predict(x)


data_with_clusters = data.copy()
data_with_clusters['Cluster'] = identify_cluster

print('Clusters based on Latitude and Longitude with K = k')
print(data_with_clusters)

#Let's plot this data using cluster variable
plt.scatter(data_with_clusters['Longitude'], data_with_clusters['Latitude'], c=data_with_clusters['Cluster'], cmap='rainbow')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.show()

