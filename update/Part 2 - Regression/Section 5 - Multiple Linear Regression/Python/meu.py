# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:26:57 2024

@author: rcastano
"""

import pandas as pd
import numpy as np
dataset =  pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


from sklearn.impute import SimpleImputer
SimpleImputer = SimpleImputer(missing_values = 0, strategy = "mean")
SimpleImputer.fit(X[:,0:3])
X[:, 0:3] = SimpleImputer.transform(X[:, 0:3])


# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#eliminar dummy que sobra (la trampa)
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#comparar amb la variable y_test
y_pred = regressor.predict(X_test)

