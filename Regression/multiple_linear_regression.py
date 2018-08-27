#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 12:38:10 2018

@author: sebastianballesteros
"""

#importing the libraries 
import numpy as np #contains mathematical tools
import matplotlib.pyplot as plt #help us plot nice charts
import pandas as pd #best library to import and manage data sets 

#Importing the dataset
dataset = pd.read_csv("50_Startups.csv")
#create matrix with independent variables (X) and dependent variables (Y)
X = dataset.iloc[:, :-1].values # ":" means all the rows, ":-1" means all columns except last one (Which is the dependent variable)
Y = dataset.iloc[:, 4].values #select all rows and just the [3] column

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  #class, OneHotEncoder for dummy encoding
label_encoder_X = LabelEncoder()
X[:, 3] = label_encoder_X.fit_transform(X[:, 3]) #returns the first column of matrix X encoded (text to numbers)
one_hot_encoder = OneHotEncoder(categorical_features = [3]) #specify which column 
X = one_hot_encoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap 
X = X[:,1:] #removing the first column of X 

#Splitting the data set into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) #independent and independent and test size (20%) as parameters

#Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train) #fit to training set 

#Predicting the test results 
Y_pred = regressor.predict(X_test)

#Building the optimal model using backward elimination
import statsmodels.formula.api as sm
X = np.append( arr = np.ones((50, 1)).astype(int), values = X, axis = 1) #you want to add a column of ones at the beginning of our matrix for statistical purposes(b0)


#Manual Backward Elimination
X_opt= X[:, [0,1,2,3,4,5]] #X_opt filled with all possible variables
regressor_OLS = sm.OLS(endog = Y, exog = X_opt ).fit() #fit the model with all possible variables
regressor_OLS.summary() #statistical summary. (ie. P values of each variable)
#we saw that the second variable was the one with the highest P value, so delete it. 
X_opt= X[:, [0,1,3,4,5]] 
regressor_OLS = sm.OLS(endog = Y, exog = X_opt ).fit() #fit the model with all possible variables
regressor_OLS.summary() 
#we saw that the first variable was the one with the highest P value, so delete it. 
X_opt= X[:, [0,3,4,5]] 
regressor_OLS = sm.OLS(endog = Y, exog = X_opt ).fit() #fit the model with all possible variables
regressor_OLS.summary() 
#we saw that the fourth variable was the one with the highest P value, so delete it. 
X_opt= X[:, [0,3,5]] 
regressor_OLS = sm.OLS(endog = Y, exog = X_opt ).fit() #fit the model with all possible variables
regressor_OLS.summary() 
#we saw that the fifth variable was the one with the highest P value, so delete it. 
X_opt= X[:, [0,3]] 
regressor_OLS = sm.OLS(endog = Y, exog = X_opt ).fit() #fit the model with all possible variables
regressor_OLS.summary() 
#Now the highest P value is 0.0000~ which is less than the significant level, so npw X_opt is the optimal model 



#Automatic Backward Elimination
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)