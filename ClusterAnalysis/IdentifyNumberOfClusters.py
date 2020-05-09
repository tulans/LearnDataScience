import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
from sklearn.cluster import KMeans

data = pd.read_csv('../data/cluster-analysis/CountryClusters.csv')
data_mapped = data.copy()
data_mapped['Language'] = data_mapped['Language'].map({'English':0, 'French':1, 'German':2})

print(data_mapped.head())
x = data_mapped.iloc[:, 3:4]

k = 3;
kmeans = KMeans(k)
identified_cluster = kmeans.fit_predict(x)
data_mapped['Cluster'] = identified_cluster

print(data_mapped)

#plt.scatter(data_mapped['Longitude'], data_mapped['Latitude'], c=data_mapped['Cluster'], cmap='rainbow')
#plt.xlim(-180, 180)
#plt.ylim(-90, 90)
#plt.show()

print('Inertia WCSS : '+str(kmeans.inertia_))

wcss=[]

for i in range(1,3):
    km = KMeans(i)
    km.fit(x)
    wcss_iter = km.inertia_
    wcss.append(wcss_iter)

print(wcss)

number_clusters = range(1,3)
plt.plot(number_clusters, wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Within cluster sum of squares')
plt.show()