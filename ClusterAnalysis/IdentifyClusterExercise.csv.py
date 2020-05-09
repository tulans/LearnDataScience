import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

data = pd.read_csv('../data/cluster-analysis/CountriesExerciseCluster.csv')
print(data.head())

x = data.iloc[:, 1:2]
from sklearn.cluster import KMeans

wcss = []

for i in range(1,15):
    km = KMeans(i)
    km.fit(x)
    wcss_iter = km.inertia_
    wcss.append(wcss_iter)

number_clusters = range(1,15)
plt.plot(number_clusters, wcss)
plt.title('Elbow method')
plt.xlabel('Numbeer of clusters')
plt.ylabel('within cluster sum of squares')
plt.show()

from sklearn.cluster import KMeans
#Let's take Lat, Long and Language to perform cluster analysis
k=10
kmeans = KMeans(k)
print('Trying '+str(k)+' means cluster ')
identified_cluster = kmeans.fit_predict(x)

data['Cluster'] = identified_cluster
plt.scatter(data['Longitude'], data['Latitude'], c=data['Cluster'], cmap='rainbow')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.show()