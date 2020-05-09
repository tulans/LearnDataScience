import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

sns.set()

raw_data = pd.read_csv('./data/linear-regression/real_estate_price_size_year_view.csv')

data = raw_data.copy()

data['view'] = data['view'].map({'No sea view':0, 'Sea view':1})

y = data['price']
x1 = data[['size', 'view']]

x = sm.add_constant(x1)

results = sm.OLS(y,x).fit()

print(results.summary())

plt.scatter(data['size'], y, c=data['view'], cmap='RdYlGn_r')
yhat_view =  7.748e+04  + 218.7521*data['size'] + 5.756e+04
yhat_no_view = 7.748e+04  + 218.7521 *data['size']
fig = plt.plot(data['size'], yhat_no_view, lw=2, c='red')
fig = plt.plot(data['size'], yhat_view, lw=2, c='orange')
#fig = plt.plot(data['size'], yhat, lw =2, c='yellow')
plt.xlabel('size', fontsize = 20)
plt.ylabel('price', fontsize = 20)
plt.show()

#Predict price basis the size and view
new_data = pd.DataFrame({'const':1, 'size':[789, 560, 934, 1045], 'view': [0, 1, 1, 0]})
new_data = new_data[['const', 'size', 'view']]
predictions = results.predict(new_data)

predictionsdf = pd.DataFrame({'Predictions': predictions})
joined = new_data.join(predictionsdf)
print(joined)
