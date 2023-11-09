#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets


# In[2]:


#load data
datasets.load_iris()


# In[9]:


#Spliting data and shaping
iris = datasets.load_iris()
x_data = iris.data
y_data = iris.target

from sklearn import model_selection

x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data, train_size = 80, random_state = 20)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[12]:


#Performing model selection using random forest classifier and fitting the model
from sklearn import ensemble

forest = ensemble.RandomForestClassifier()

forest.fit(x_train, y_train)


# In[14]:


#Performing model selection using linear regression and fitting the model

from sklearn import linear_model

mod = linear_model.LinearRegression()

mod.fit(x_train, y_train)


# In[21]:


#Performing model selection using SVM regression and fitting the model

from sklearn import svm

sup = svm.SVR()

sup.fit(x_train, y_train)


# In[22]:


#creating y-predictions to perfrom metrics
forest_preds = forest.predict(x_test)
forest_preds[forest_preds < 0] = 0

sup_preds = sup.predict(x_test)
sup_preds[sup_preds < 0] = 0

mod_preds = mod.predict(x_test)
mod_preds[mod_preds < 0] = 0

from sklearn import metrics

forest_variance = metrics.explained_variance_score(y_test, forest_preds)

forest_mean_squared = metrics.mean_squared_error(y_test, forest_preds)

forest_r_squared = metrics.r2_score(y_test, forest_preds)


mod_variance = metrics.explained_variance_score(y_test, mod_preds)

mod_mean_squared = metrics.mean_squared_error(y_test, mod_preds)

mod_r_squared = metrics.r2_score(y_test, mod_preds)


sup_variance = metrics.explained_variance_score(y_test, sup_preds)

sup_mean_squared = metrics.mean_squared_error(y_test, sup_preds)

sup_r_squared = metrics.r2_score(y_test, sup_preds)
 
print("The random forest regression variance score is:", forest_variance)
print("\nThe random forest regression mean squared error is:", forest_mean_squared)
print("\nThe random forest r^2 error is:", forest_r_squared)

print("\nThe linear regression variance score is:", mod_variance)
print("\nThe linear regression mean squared error is:", mod_mean_squared)
print("\nThe linear regression r^2 error is:", mod_r_squared)

print("\nThe vector machine regression variance score is:", sup_variance)
print("\nThe vector machine regression mean squared error is:", sup_mean_squared)
print("\nThe vector machine regression r^2 error is:", sup_r_squared)


# The linear regression model is the better performer between the SVM regression and the random forest regression models. The linear regression shows that it has a better fit for the data because it has the highest r^2 value. This model also shows that it has fewer prediction errors due to the linear regression model having the lowest mean squared value.

# In[33]:


from scipy import interpolate
import numpy as np


# In[38]:


#Extract Sepal length and Width
sepal_length = iris.data[:, 0]
sepal_width = iris.data[:, 1]

x_train, x_test, y_train, y_test = model_selection.train_test_split(sepal_length, sepal_width, test_size=15, train_size=135, random_state=0)

#Perform interpolation
model = interpolate.interp1d(x_train, y_train, kind='linear')

#Calculate y-prediction for interpolation
y_pred_interp = model(x_test)

#performing linear regression
regression_model = linear_model.LinearRegression()
regression_model.fit(x_train.reshape(-1,1), y_train)

#Calculate y-prediction for linear regression
y_pred_regression = regression_model.predict(x_test.reshape(-1,1))

#Calculating absolute difference
abs_diff_interp = np.abs(y_test - y_pred_interp)
abs_diff_regression = np.abs(y_test - y_pred_regression)

#Calculating the sum of abs. differences
sum_abs_diff_interp = np.sum(abs_diff_interp)
sum_abs_diff_regression = np.sum(abs_diff_regression)

print(f"Sum of Absolute Differences for interpolation:", sum_abs_diff_interp)
print(f"Sum of Absolute Differences for linear regression:", sum_abs_diff_regression)


# Based on my results, interpolation was the better because it has a lower absolute difference value than the linear regression approximation.
