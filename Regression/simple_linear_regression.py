#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 17:15:38 2018

@author: sebastianballesteros
"""

#SIMPLE LINEAR REGRESSION

#Data Preprocessing

#importing the libraries 
import numpy as np #contains mathematical tools
import matplotlib.pyplot as plt #help us plot nice charts
import pandas as pd #best library to import and manage data sets 

#Importing the dataset
dataset = pd.read_csv("Salary_Data.csv")
#create matrix with independent variables (X) and dependent variables (Y)
X = dataset.iloc[:, :-1].values # ":" means all the rows, ":-1" means all columns except last one (Which is the dependent variable)
Y = dataset.iloc[:, 1].values #select all rows and just the [1] column

#Splitting the data set into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0) #independent and independent and test size (33.33%) as parameters

#fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train) #the regressor is the machine we are creating 

#predicting the Test set results
Y_pred = regressor.predict(X_test) #now compare y_pred to y_test

#Visualizing the training set results 
plt.scatter(X_test, Y_test, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()