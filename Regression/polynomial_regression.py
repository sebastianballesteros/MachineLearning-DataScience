#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 16:24:42 2018

@author: sebastianballesteros
"""

#POLYNOMIAL REGRESSION 

#importing the libraries 
import numpy as np #contains mathematical tools
import matplotlib.pyplot as plt #help us plot nice charts
import pandas as pd #best library to import and manage data sets 

#Importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
#create matrix with independent variables (X) and dependent variables (Y)
X = dataset.iloc[:, 1:2].values # upper bound is excluded
Y = dataset.iloc[:, 2].values #select all rows and just the [3] column

#Fitting Linear Regression to the data set
from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(X, Y)


#Fitting Polynomial Regression to the data set
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) #transforms to a new matrix composed of x^n 
X_poly = poly_reg.fit_transform(X) #variable with its associated polynomial terms
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y) #build a linear regression with the two independent variables, the position level, and the position level squared

#Visualizing the LinearRegression results
plt.scatter(X, Y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Truth or Bluff (Linear R)")
plt.xlabel("Position Label")
plt.ylabel("Salary")
plt.show()

#Visualizing the Polynomial Regression reuslts
plt.scatter(X, Y, color = "red")
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = "blue")
plt.title("Truth or Bluff (Polynomial R)")
plt.xlabel("Position Label")
plt.ylabel("Salary")
plt.show()

#Predicting a new Result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))
