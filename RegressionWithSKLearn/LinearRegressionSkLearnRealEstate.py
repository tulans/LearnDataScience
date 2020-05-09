import pandas as pd
import seaborn as sns
import sklearn as sk
import matplotlib.pyplot as plt
sns.set()

data = pd.read_csv('../data/linear-regression/real_estate_price_size.csv')
from sklearn.linear_model import LinearRegression

print(data.head())

x = data['size']
y = data['price']

x_matrix = x.values.reshape(-1,1)

reg = LinearRegression()

reg.fit(x_matrix, y)

#1. R Square value
print(reg.score(x_matrix, y))

#2. Co-efficient
print(reg.coef_)

#3. Intercept
print(reg.intercept_)

#4. Predict Price
new_data = pd.DataFrame(data=[789, 890], columns=['size'])
new_data['Predicted_Price'] = reg.predict(new_data)

print(new_data)
plt.scatter(x,y)
yhat = reg.coef_ * x_matrix + reg.intercept_
fig = plt.plot(x, yhat, lw=2, c='red', label='regression line')
plt.xlabel('size', fontsize=15)
plt.ylabel('price', fontsize=15)
plt.show()