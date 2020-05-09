import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
#Multiple linear regression model taking into consideration one more variable i.e random
# GPA = b0 + b1*SAT + b2*Rand
sns.set()
data = pd.read_csv('./data/linear-regression/real_estate_price_size_year.csv')
print(data.describe())

y = data['price']
x2 = data['size']
x3 = data['year']
x1 = data[['size', 'year']]


x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()

print(results.summary())

#yhat = -5.772e+06 + 227.7009*x2 + 2916.7853*x3
#fig = plt.plot(x1, yhat, lw=4, c='red', label='regression line')
#plt.xlabel('SIZE AND YEAR', fontsize = 20)
#plt.ylabel('PRICE', fontsize = 20)
#plt.show()