import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
raw_data = pd.read_csv('../data/cluster-analysis/Countryclusters.csv')
print(raw_data)

data = raw_data.copy()
data['Language'] = data['Language'].map({'English':0, 'French':1, 'German':2})

from sklearn.cluster import KMeans
#Let's take Lat, Long and Language to perform cluster analysis
x = data.iloc[:, 1:4]
k = 3
kmeans = KMeans(k)
print('Trying '+str(k)+' means cluster ')
identified_cluster = kmeans.fit_predict(x)

data['Cluster'] = identified_cluster
plt.scatter(data['Longitude'], data['Latitude'], c=data['Cluster'], cmap='rainbow')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.show()

