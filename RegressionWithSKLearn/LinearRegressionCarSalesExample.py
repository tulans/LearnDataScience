import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()

raw_data = pd.read_csv('../data/linear-regression/CarSalesExampleForRegression.csv')
print(raw_data.head())

print(raw_data.describe(include='all'))

data = raw_data.drop(['Model'], axis=1)

print(data.isnull().sum())
#Price and EngineVolumen are having many null entities

data_no_mv = data.dropna(axis=0)

print(data_no_mv.describe(include='all'))

sns.distplot(data_no_mv['Price'])

q = data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price']<q]

print(data_1.describe(include='all'))

sns.distplot(data_1['Price'])


#Let's check mileage and clean up the data
sns.distplot(data_1['Mileage'])

qm = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage'] < qm]

sns.distplot(data_2['Mileage'])

data_3 = data_2[data_2['EngineV']<6.5]
sns.distplot(data_3['EngineV'])
plt.show()

year = data_3['Year'].quantile(0.01)
data_4 = data_3[data_3['Year']>year]

data_cleaned = data_4.reset_index(drop=True)

print(data_cleaned.describe(include='all'))


#All the graphs plotted are not having linear regression right now. Hence need to find
# or transform the plots in logarithmic manner
log_price = np.log(data_cleaned['Price'])
data_cleaned['Log_Price'] = log_price
print(data_cleaned)

f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, sharey=True, figsize=(15, 6))
ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['Mileage'], data_cleaned['Price'])
ax2.set_title('Mileage and Price')
ax3.scatter(data_cleaned['EngineV'], data_cleaned['Price'])
ax3.set_title('EngineV and Price')
ax4.scatter(data_cleaned['Year'], data_cleaned['Log_Price'])
ax4.set_title('Log Price and Year')
ax5.scatter(data_cleaned['Mileage'], data_cleaned['Log_Price'])
ax5.set_title('Mileage and Log Price')
ax6.scatter(data_cleaned['EngineV'], data_cleaned['Log_Price'])
ax6.set_title('EngineV and Log Price')
plt.show()

#check for no endogeneity test
#Log transformation is the common fix for Heteroscedascticity.
print(data_cleaned.columns.values)

#SKLearn doesn't have a method to check for multi-colinearity. Hence rely on
#stats model to identify the multi-colienarity.
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage', 'Year', 'EngineV']]
vlf = pd.DataFrame()
vlf['VLF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vlf['features'] = variables.columns
print(vlf)

#VLF:
# if vlf = 1, then No multicollinearity
# if 1 < vlf 5 then perfectly okay
# if vlf > 6 then unacceptable

#as year vlf value is greater than 10 and hence can be dropped or ignored
data_no_multicollinearity = data_cleaned.drop(['Year'], axis=1)

#if we include a separate dummy variable for each category, we will introduce multicollinearity
#to the regression.
data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)

target = data_with_dummies['Log_Price']
inputs = data_with_dummies.drop(['Log_Price'], axis=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(inputs)
input_scaled = scaler.transform(inputs)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(input_scaled, target, test_size=0.2, random_state=365)

reg = LinearRegression()
reg.fit(x_train, y_train)

y_hat = reg.predict(x_train)

plt.scatter(y_train, y_hat)
plt.xlabel('Targets (y_train)', size=18)
plt.ylabel('Targets (y_hat)', size=18)
plt.xlim(6, 13)
plt.ylim(6, 13)
plt.show()

sns.distplot(y_train - y_hat)
plt.title('Residual PDF ', size=18)
plt.show()

print(reg.score(x_train, y_train))

reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
print(reg_summary)
print(data_cleaned['Brand'].unique())

#Weights Interpretation
#1. continuous variables
#A posiitive wieght shows that as a feature increases in value so do the log_price and price respectively
#A negative weight showss that as a feature increases in values login_price and price decreases
#2. Dummy variables
#1. Positive weight shows that respective category is more expensive than the benchmark


#Testing:
y_hat_test = reg.predict(x_test)
plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('Targets (y_train)', size=18)
plt.ylabel('Targets (y_hat_test)', size=18)
plt.xlim(6, 13)
plt.ylim(6, 13)
plt.show()


df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Predictions'])
df_pf['Target'] = np.exp(y_test)

print(df_pf.head())

y_test = y_test.reset_index(drop=True)
print(y_test.head())
df_pf['Target'] = np.exp(y_test)

print(df_pf.head())

df_pf['Residual'] = df_pf['Target'] - df_pf['Predictions']
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
print(df_pf.describe())

print(df_pf.sort_values(by=['Difference%']))





