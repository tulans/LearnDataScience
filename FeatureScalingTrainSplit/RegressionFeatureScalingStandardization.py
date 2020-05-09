import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
data = pd.read_csv('./data/linear-regression/1.02.Multiple_linear_regression.csv')
print(data.head())

x=data[['SAT','Rand 1,2,3']]
y=data['GPA']

from sklearn.linear_model import LinearRegression
#without standard scaler
old_reg = LinearRegression()
old_reg.fit(x, y)

old_reg_summary = pd.DataFrame([['intercept'], ['SAT'], ['Rand 1,2,3']], columns=['Features'])
old_reg_summary['weights'] = old_reg.intercept_, old_reg.coef_[0], old_reg.coef_[1]
print('Before standardization i.e before transforming inputs')
print(old_reg_summary)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#Whenever feature scaling needs to be done i.e standardization then here is the formulae for the same
# standardized variable = (x - mu)/signma
#where x is original variable
# mu is mean of original variable
#standard deviation of original variable


#Scaling mechanism
scaler.fit(x)

#Transform the unscaled input to scaled inputs
x_scaled = scaler.transform(x)

#Regression with scaled inputs on all features.
#Feature selection with scaled inputs


reg = LinearRegression()
reg.fit(x_scaled, y)

reg_summary = pd.DataFrame([['bias'], ['SAT'], ['Rand 1,2,3']], columns=['Features'])
#Weights is the machine language terminology for coefficients. Bigger the weight
#machine language term for intercept is bias
reg_summary['weights'] = reg.intercept_, reg.coef_[0], reg.coef_[1]
print('After Standardization i.e transaforming inputs')
print(reg_summary)
#SK Learn natively doesn't support p values since all the features being used for regression
#are feature scaled with standardization of weights
#After performing feature scaling, we don't care if a useless variable is there or not.

#Making predictions with the standadized co-efficients (weights)
new_data = pd.DataFrame(data=[[1700,2], [1000,1]], columns=['SAT', 'Rand 1,2,3'])
new_scaled_data = scaler.transform(new_data)
print('predicted data set with both SAT and Rand 1,2,3, features '+str(reg.predict(new_scaled_data)))


#Let's check the predicted data set if the feature set contained only SAT score and not the random
#variable or feature
reg_simple = LinearRegression()
x_simple_matrix = x_scaled[:,0].reshape(-1, 1)
reg_simple.fit(x_simple_matrix, y)
print('Result set with only one feature i.e SAT score ' + str(reg_simple.predict(new_scaled_data[:, 0].reshape(-1, 1))))





