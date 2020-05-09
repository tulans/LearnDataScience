import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

data = pd.read_csv('./data/linear-regression/AddDummiesForGrade.csv');

#print(data.describe())

new_data = data.copy()
new_data['Attendance'] = data['Attendance'].map({'Yes':1, 'No':0})

print(new_data.describe())

y = new_data['GPA']
x1 = new_data[['SAT', 'Attendance']]

x = sm.add_constant(x1)

results = sm.OLS(y,x).fit()
print(results.summary())

plt.scatter(new_data['SAT'],y, c=new_data['Attendance'], cmap='RdYlGn_r')
yhat_no = 0.6439 + 0.0014*new_data['SAT']
yhat_yes = 0.8665 + 0.0014*new_data['SAT']
yhat = 0.0017*new_data['SAT'] + 0.275
fig = plt.plot(new_data['SAT'], yhat_no, lw=2, c='red')
fig = plt.plot(new_data['SAT'], yhat_yes, lw=2, c='orange')
fig = plt.plot(new_data['SAT'], yhat, lw =2, c='yellow')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()
