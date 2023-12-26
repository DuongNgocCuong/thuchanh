# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 09:02:08 2023

@author: USER
"""
#Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
print('Libraries Imported')
#Creating Dataset and including the first row by setting no header as input
dataset = pd.read_csv('iris.csv', header = None)
#Renaming the columns
dataset.columns = ['sepal length in cm', 'sepal width in cm','petal length in cm','petal width in cm','species']
print('Shape of the dataset: ' + str(dataset.shape))
print(dataset.head())
#Creating the dependent variable class
factor = pd.factorize(dataset['species'])
dataset.species = factor[0]
definitions = factor[1]
print(dataset.species.head())
print(definitions)
#Splitting the data into independent and dependent variables
X = dataset.iloc[:,0:4].values
y = dataset.iloc[:,4].values
print('The independent features set: ')
print(X[:5,:])
print('The dependent variable: ')
print(y[:5])

# Creating the Training and Test set from data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 21)