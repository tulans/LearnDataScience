import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
data = pd.read_csv('../data/linear-regression/real_estate_price_size_year.csv')
print(data.head())
print(data.describe())
x = data[['size', 'year']]
y = data['price']
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x,y)

# Let's use the handy function we created
def adj_r2(x,y):
    r2 = reg.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2

adjusted_r2_value = adj_r2(x, y)
print(adjusted_r2_value)
print('predicted value '+str(reg.predict([[750,2009]])))

from sklearn.feature_selection import f_regression
f_regression(x,y)
p_values = f_regression(x, y)[1]
new_p = p_values.round(3)

reg_summary = pd.DataFrame(data = x.columns.values, columns=['Features'])
reg_summary['Coefficients'] = reg.coef_
reg_summary['p-values'] = new_p

print(reg_summary)
