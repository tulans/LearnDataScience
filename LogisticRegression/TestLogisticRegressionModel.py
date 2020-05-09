import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
sns.set()

raw_data = pd.read_csv('../data/logistic-regression/Binarypredictors.csv')
data = raw_data.copy()
print(data.head())
data['Admitted'] = data['Admitted'].map({'Yes':1, 'No':0})
data['Gender'] = data['Gender'].map({'Female':1, 'Male':0})
y = data['Admitted']
x1 = data[['SAT', 'Gender']]

x = sm.add_constant(x1)
reg_log = sm.Logit(y,x)
results_log = reg_log.fit()
print(results_log.summary())


#Accuracy
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
#np.set_printoptions(formatter=None)
print(results_log.predict())
np.array(data['Admitted'])
results_log.pred_table()

cm_df = pd.DataFrame(results_log.pred_table())
cm_df.columns = ['Predicted 0', 'Predicted 1']
cm_df = cm_df.rename(index={0: 'Actual 0',1:'Actual 1'})
print(cm_df)

#Load Test data set:

test = pd.read_csv('../data/logistic-regression/Testdataset.csv')
# Map the test data as you did with the train data
test['Admitted'] = test['Admitted'].map({'Yes': 1, 'No': 0})
test['Gender'] = test['Gender'].map({'Female': 1, 'Male': 0})

# Get the actual values (true valies ; targets)
test_actual = test['Admitted']
# Prepare the test data to be predicted
test_data = test.drop(['Admitted'],axis=1)
test_data = sm.add_constant(test_data)


def confusion_matrix(data, actual_values, model):
    # Confusion matrix

    # Parameters
    # ----------
    # data: data frame or array
    # data is a data frame formatted in the same way as your input data (without the actual values)
    # e.g. const, var1, var2, etc. Order is very important!
    # actual_values: data frame or array
    # These are the actual values from the test_data
    # In the case of a logistic regression, it should be a single column with 0s and 1s

    # model: a LogitResults object
    # this is the variable where you have the fitted model
    # e.g. results_log in this course
    # ----------

    # Predict the values using the Logit model
    pred_values = model.predict(data)
    # Specify the bins
    bins = np.array([0, 0.5, 1])
    # Create a histogram, where if values are between 0 and 0.5 tell will be considered 0
    # if they are between 0.5 and 1, they will be considered 1
    cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]
    # Calculate the accuracy
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    # Return the confusion matrix and the accuracy
    return cm, accuracy

# Create a confusion matrix with the test data
cm = confusion_matrix(test_data,test_actual,results_log)
print(cm)

# Format for easier understanding (not needed later on)
cm_df = pd.DataFrame(cm[0])
cm_df.columns = ['Predicted 0','Predicted 1']
cm_df = cm_df.rename(index={0: 'Actual 0',1:'Actual 1'})
print(cm_df.describe())


# Check the missclassification rate
# Note that Accuracy + Missclassification rate = 1 = 100%
print ('Missclassification rate: '+str((1+1)/19))


