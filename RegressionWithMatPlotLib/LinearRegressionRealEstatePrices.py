import matplotlib.pyplot as plt
import numpy as nm
import pandas as pd
import statsmodels.api as sm
import seaborn as sns

sns.set()


data = pd.read_csv('./data/linear-regression/real_estate_price_size.csv')
print(data)
print(data.describe())
x1 = data['size']
y = data['price']

plt.scatter(x1, y)
plt.xlabel('size', fontsize = 20)
plt.ylabel('price', fontsize = 20)
#plt.show()

x = sm.add_constant(x1)

results  = sm.OLS(y,x).fit()

print(results.summary())

yhat = 223.1787*x1 + 1.19e+05
plt.plot(x1, yhat, lw =4, c='red', label='regression line')
plt.xlabel('size', fontsize = 20)
plt.ylabel('price', fontsize = 20)
plt.show()