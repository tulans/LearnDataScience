import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set()
data = pd.read_csv('../data/linear-regression/linear-regression.csv')
print(data.head())

from sklearn.linear_model import LinearRegression

x = data['SAT']
y = data['GPA']
print(x.shape)
x_matrix = x.values.reshape(-1, 1)
print(x_matrix.shape)
reg = LinearRegression()

reg.fit(x_matrix, y)

#simple linear regression summary same as statsmodel that we had seen earlier

#1.R Squared
print(reg.score(x_matrix, y))

#2. Coefficients
print(reg.coef_)

#3. intercept
print(reg.intercept_)

#4. Predict GPA
new_data = pd.DataFrame(data=[1740, 1760], columns=['SAT'])
new_data['Predicted_GPA'] = reg.predict(new_data)
print(new_data)

plt.scatter(x,y)
yhat = reg.coef_*x_matrix + reg.intercept_
fig = plt.plot(x, yhat, lw=4, c='orange', label='regression line')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()