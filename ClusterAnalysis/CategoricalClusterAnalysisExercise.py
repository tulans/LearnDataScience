import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
sns.set()
from sklearn.cluster import KMeans
raw_data = pd.read_csv('../data/cluster-analysis/CategoricalExercise.csv')
print('Raw Data')
print(raw_data)

data = raw_data.copy()
print('Unique values for continent '+str(data.continent.unique()))
data['continent'] = data['continent'].map(
    {'North America': 0,
     'Europe': 1,
     'Africa': 2,
     'Asia': 3,
     'Oceania': 5,
     'South America': 6,
     'Antarctica': 7,
     'Seven seas (open ocean)': 8
     })

x = data.iloc[:, 3:4]
print(x)
k = 10
print('Trying with '+str(k) + ' clustering solution')
kmeans = KMeans(k)
identified_cluster = kmeans.fit_predict(x)

data['Cluster'] = identified_cluster
plt.scatter(data['Longitude'], data['Latitude'], c=data['Cluster'], cmap='rainbow')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.show()
