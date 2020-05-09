import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set()

data = pd.read_csv('../data/linear-regression/1.02.Multiple_linear_regression.csv')

print(data.head())
print(data.describe())
from sklearn.linear_model import LinearRegression


x = data[['SAT', 'Rand 1,2,3']]
y = data['GPA']

reg = LinearRegression()
reg.fit(x,y)

#1. R Score value
# Same function is used for both multiple and linear regression
print('R Squared Score ' + str(reg.score(x,y)))

#Find Adjusted R score. SK Learn library doesn't have the ability to identify r sqaured
r2 = reg.score(x, y)
n = x.shape[0]
p = x.shape[1]
adjusted_r2 = 1 - (1-r2)*(n-1)/(n-p-1)

if(r2 > adjusted_r2):
    print('Some of the attributes used have little or no explanatory power since adjusted R2 is less than R Squared')

#Identify pvalue for each of the features used in the linear regression model
# using sk learn. We had seen this as part of statsmodel.
#we know - if the pvalue is > 0.05 then we can disregard that feature.
#There is no direct method to get the pvalue but SK Learn provides a method to find the
#F Stats i.e Feature selection based on the linear regression model for each of the feature
#Example - impact of SAT on GPA
#Example - impact of Rand 1,2,3 on GPA

from sklearn.feature_selection import f_regression
print('Feature selection stats for the features (Fstats Array + PValues Array) - '+str(f_regression(x,y)))

#output is in scientific notations - e-11 === * 10 to the power of -11 = /10 to the power of 11
p_values = f_regression(x,y)[1]
print('PValues ' + str(p_values))

new_p_values = p_values.round(3)
print('Pvalues rounded '+str(new_p_values))

#Note that - these are univariate p-values reached or derived from simple linear models
#They do not reflect the interconnection of the features in our multiple linear regression
reg_summary = pd.DataFrame(data=x.columns.values, columns=['Features'])

reg_summary['Coefficients'] = reg.coef_
reg_summary['p-values'] = p_values.round(3)
print(reg_summary)

#2. Coefficient
print('Coefficient Arrays ' + str(reg.coef_))

#3. Intercept
print('Intercept or constant ' + str(reg.intercept_))

