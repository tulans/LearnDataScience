import numpy as nm
import pandas as pd
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl

# each time a regression model is created. it should be meaningful.
# y = b0 + b1x1
data = pd.read_csv('./data/linear-regression/linear-regression.csv')

y = data['GPA']
x1 = data['SAT']

plt.scatter(x1,y)
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)
#plt.show()

x = sm.add_constant(x1)

results = sm.OLS(y,x).fit()

print(results.summary())

#based on result summary of linear regression models - pick the right coef and
#print(data.describe())

yhat = 0.0017*x1 + 0.2750
fig = plt.plot(x1, yhat, lw=4, c='red', label='regression line')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()


