#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:19:38 2018

@author: sebastianballesteros

Data Preprocessing Template

"""

#importing the libraries 
import numpy as np #contains mathematical tools
import matplotlib.pyplot as plt #help us plot nice charts
import pandas as pd #best library to import and manage data sets 

#Importing the dataset
dataset = pd.read_csv("Data.csv")
#create matrix with independent variables (X) and dependent variables (Y)
X = dataset.iloc[:, :-1].values # ":" means all the rows, ":-1" means all columns except last one (Which is the dependent variable)
Y = dataset.iloc[:, 3].values #select all rows and just the [3] column

"""
#Taking care of missing data 
from sklearn.preprocessing import Imputer #class
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0) #create Imputer object
imputer = imputer.fit(X[:, 1:3]) #fit the mean in the missing values (columns) upper bound is excluded 1-2
X[:, 1:3] = imputer.transform(X[:, 1:3])
"""

"""
#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  #class, OneHotEncoder for dummy encoding
label_encoder_X = LabelEncoder()
X[:, 0] = label_encoder_X.fit_transform(X[:, 0]) #returns the first column of matrix X encoded
one_hot_encoder = OneHotEncoder(categorical_features = [0]) #specify which column 
X = one_hot_encoder.fit_transform(X).toarray()
label_encoder_Y = LabelEncoder() 
Y = label_encoder_Y.fit_transform(Y) #for the dependent variable (Yes, No) we don't need OneHotEncoder
""" 

#Splitting the data set into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) #independent and independent and test size (20%) as parameters

"""
#Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) #when you are applying object to a training set you have to fit object and then transform it 
X_test = sc_X.transform(X_test) #we just need to transform it cause its already transformed by training set 
"""