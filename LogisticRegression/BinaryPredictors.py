import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
sns.set()

raw_data = pd.read_csv('../data/logistic-regression/Binarypredictors.csv')
data = raw_data.copy()
data['Admitted'] = data['Admitted'].map({'Yes':1, 'No': 0})
data['Gender'] = data['Gender'].map({'Female':1, 'Male':0})

y = data['Admitted']
x1 = data['Gender']

x = sm.add_constant(x1)
reg_log = sm.Logit(y, x)
results_log = reg_log.fit()

print(results_log.summary())

#log(odds) = -0.64 + 2.08 * Gender  <-- This is based on Summary
# log(odds1) = -0.64 + 2.08 * Gender1
# log(odds2) = -0.64 + 2.08 * Gender2
#log (odds2/odds1) = 2.08 * (Gender2 - Gender1)

# Now since Gender values is either female or male
# log(odds_female/odds_male) = 2.08 * (1 -0 )
#And thus
#odds_female = np.exp(2.08) * odds_male
print(np.exp(2.0786))


#Now let's try with feature set that includes both Gender and SAT
x2 = data[['Gender', 'SAT']]
x_new = sm.add_constant(x2)
reg_log_new = sm.Logit(y, x_new)
results_log_new = reg_log_new.fit()

print(results_log_new.summary())
print('Female has '+str(np.exp(1.9449)) +' times higher odds to get admitted as compared to male')
