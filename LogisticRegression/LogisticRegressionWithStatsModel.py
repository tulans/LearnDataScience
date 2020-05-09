import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
sns.set()

raw_data = pd.read_csv('../data/logistic-regression/Admittance.csv')
data = raw_data.copy()

data['Admitted'] = data['Admitted'].map({'Yes':1, 'No':0})

y = data['Admitted']
x1 = data['SAT']

x = sm.add_constant(x1)
reg_log = sm.Logit(y, x)
results_log = reg_log.fit()

print(results_log.summary())