



import pandas as pd
LabourData = pd.read_csv(r'LabourEarningPrediction.csv')
LabourData.head()    

LabourData.tail()

LabourData.info()
LabourData.describe()

import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline


sns.pairplot(LabourData)




sns.pairplot(LabourData, x_vars=['Age','Earnings_1974','Earnings_1975'], y_vars='Earnings_1978',
             size=7, aspect=0.7, kind='scatter')

sns.boxplot(data=LabourData["Age"])

sns.boxplot( x=LabourData["Race"], y=LabourData["Earnings_1978"] )


sns.violinplot( x=LabourData["Hisp"], y=LabourData["Earnings_1978"] )

LabourData_num = LabourData[['Age','Nodeg', 'Earnings_1974', 'Earnings_1975', 'Earnings_1978']]


LabourData_dummies = pd.get_dummies(LabourData[['Race', 'Hisp', 'MaritalStatus', 'Eduacation']])

LabourData_dummies.head()

LabourData_combined = pd.concat([LabourData_num, LabourData_dummies], axis=1)

LabourData_combined.head()

X = LabourData_combined[['Age', 'Earnings_1974', 'Earnings_1975', 'Race_NotBlack', 'Race_black', 
                         'Hisp_NotHispanic', 'Hisp_hispanic','MaritalStatus_Married', 
                         'MaritalStatus_NotMarried', 'Eduacation_HighSchool', 'Eduacation_Intermediate',
                         'Eduacation_LessThanHighSchool', 'Eduacation_PostGraduate', 'Eduacation_graduate']]

# Putting response variable to y
y = LabourData['Earnings_1978']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 , random_state=100)


from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train,y_train)

print(lm.intercept_)


coeff_df = pd.DataFrame(lm.coef_,X_test.columns,columns=['Coefficient'])
coeff_df

y_pred = lm.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)


from math import sqrt
rmse = sqrt(mse)

print('Mean_Squared_Error :' ,mse)
print('Root_Mean_Squared_Error :' ,rmse)
print('r_square_value :',r_squared)


############################################################################


import statsmodels.api as sm
X_train_sm = X_train
X_train_sm = sm.add_constant(X_train_sm)
# create a fitted model in one line
lm_1 = sm.OLS(y_train,X_train_sm).fit()

# print the coefficients
lm_1.params

print(lm_1.summary())

import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline


plt.figure(figsize = (5,5))
sns.heatmap(LabourData_num.corr(),annot = True)


c = [i for i in range(1,150,1)]
fig = plt.figure()
plt.plot(c,y_test[1:150], color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,y_pred[1:150], color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
plt.xlabel('Labour', fontsize=18)                               # X-label
plt.ylabel('Earnings_1978', fontsize=16)                               # Y-label



c = [i for i in range(1,y_pred.shap[0],1)]
fig = plt.figure()
plt.plot(c,y_test-y_pred, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
plt.xlabel('Index', fontsize=18)                      # X-label
plt.ylabel('Actual - Predicted', fontsize=16)                # Y-label



print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


X_train_final = X_train
#Unlike SKLearn, statsmodels don't automatically fit a constant, 
#so you need to use the method sm.add_constant(X) in order to add a constant. 
X_train_final = sm.add_constant(X_train_final)
# create a fitted model in one line
lm_final = sm.OLS(y_train,X_train_final).fit()

print(lm_final.summary())

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

