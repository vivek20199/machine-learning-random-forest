#importing libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# importing dataset
data = pd.read_csv('./insurance.csv')

# findout features 
print(data.head())
print("\n\n")
print(data.shape)
print("\n\n")
# any null values?
print(data.isnull().sum())
print("\n\n")
# encoding  categorical to numerical
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data['smoker'] = le.fit_transform(data['smoker'])
data['region'] = le.fit_transform(data['region'])

data.head(5)

# assigning target variable charges to X
X = data.drop('charges',axis=1)
y = data['charges']

# 60:40 split of data into testing and training respectively
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 20)


# random forest regression model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 150,max_leaf_nodes=4, random_state = 30)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(y_pred)
print("\n\n")
print("Regression Score:\n")
print(regressor.score(X_test,y_test))
print("\n\n")

# importing metrics for calculating rmse,mae,mse
from sklearn import metrics
print('Mean Absolute Error(MAE):', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error(MSE):', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error(RMSE):', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

#plotting graph
x_ax = range(len(y_test))
plt.plot(x_ax, y_test, linewidth=1, label="original")
plt.plot(x_ax, y_pred, linewidth=1.1, label="predicted")
plt.title("y-test and y-predicted data")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show() 

# end #
