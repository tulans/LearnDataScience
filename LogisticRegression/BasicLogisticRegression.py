import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

raw_data = pd.read_csv('../data/logistic-regression/Admittance.csv')

print(raw_data.head())

data = raw_data.copy()

data['Admitted'] = data['Admitted'].map({'Yes':1, 'No':0})

print(data.head())

y = data['Admitted']
x1 = data['SAT']

#plot with logistic regression
x = sm.add_constant(x1)
reg_log = sm.Logit(y, x)
results_log = reg_log.fit()

def f(x, b0, b1):
    return np.array(np.exp(b0+x*b1)/(1+np.exp(b0+x*b1)))

f_sorted = np.sort(f(x1, results_log.params[0], results_log.params[1]))
x_sorted = np.sort(np.array(x1))


plt.scatter(x1, y, color='C0')
plt.xlabel('SAT', fontsize=10)
plt.ylabel('Admitted', fontsize=10)
plt.plot(x_sorted, f_sorted, color='C0')
plt.show()



