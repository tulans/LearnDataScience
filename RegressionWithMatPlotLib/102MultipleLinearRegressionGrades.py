import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
#Multiple linear regression model taking into consideration one more variable i.e random
# GPA = b0 + b1*SAT + b2*Rand
sns.set()
data = pd.read_csv('./data/linear-regression/1.02.Multiple_linear_regression.csv')
print(data.describe())

y = data['GPA']
x1 = data[['SAT', 'Rand 1,2,3']]

x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()

print(results.summary())