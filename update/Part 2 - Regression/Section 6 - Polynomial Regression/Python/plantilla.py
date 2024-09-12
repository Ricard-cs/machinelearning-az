# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 11:30:13 2024

@author: rcastano
"""

# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#dades que falten
#Modificar Nan per mitjanes
from sklearn.impute import SimpleImputer
SimpleImputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
SimpleImputer.fit(X[:,1:3])
X[:, 1:3] = SimpleImputer.transform(X[:, 1:3])


#Categoritzar el país cap a dades numèriques (aquest ordinal)
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
#le_X = preprocessing.LabelEncoder() #cat ordinal
#X[:,0] = le_X.fit_transform(X[:,0]) #cat ordinal
#han de ser variables dummy (vectors amb tantes posicions com categories. Pren 1 només en una posició)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
#si / no no es dummy no binary
le_Y = preprocessing.LabelEncoder() #cat ordinal
y[:] = le_Y.fit_transform(y[:]) #cat ordinal
### IMPORTANT! tenir en compte que les variables dummy, una s'ha d'elimianr per temes de dependencia 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#escalat
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit(X_test)
#també fer a Y si no és de classificació

####EXEMPLE A PART sense modificar dades

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values # ha de ser matriu
y = dataset.iloc[:, -1].values


# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
#transformacio polinomica
poly_reg = PolynomialFeatures(degree = 4) #es busca més ajust (per defecte és 2)
X_poly = poly_reg.fit(X)
X_poly = poly_reg.transform(X)
#X_poly = poly_reg.fit_transform(X) #es el mateix que les dues de dalt

#consctucció del model polinomic a partir de la transformació polinomica
from sklearn.linear_model import LinearRegression
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#prediccions
#cal fer el fit transform pels nivells de la funcio polinomica
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))


# Visualising the Polynomial Regression results i el lineal per compara
#fem el grafic amb més un punts 

x_grid = np.arange(1,10, 0.1)
x_grid = x_grid.reshape(len(x_grid),1)

plt.scatter(X, y, color = 'red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

