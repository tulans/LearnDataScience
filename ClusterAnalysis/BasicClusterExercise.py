import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

data = pd.read_csv('../data/cluster-analysis/Countries-exercise.csv')

#raw data
#plt.scatter(data['Longitude'], data['Latitude'])
#plt.xlim(-180, 180)
#plt.ylim(-90,90)
#plt.show()

print('Raw data to be analyzed')
print(data)

print('Use dataframe iloc method to get Lat,Long ')
x = data.iloc[:, 2:4]
print(x)

#Number of clusters that we want to produce is configurable here.
k = 5
print('Trying '+str(k)+'means clustering')
kmeans = KMeans(k)
identify_clusters = kmeans.fit_predict(x)

data_with_cluster = data.copy()
data_with_cluster['Cluster'] = identify_clusters

plt.scatter(data_with_cluster['Longitude'], data_with_cluster['Latitude'], c=data_with_cluster['Cluster'], cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90, 90)
plt.show()





