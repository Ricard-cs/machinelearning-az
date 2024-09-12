# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:28:00 2024

@author: rcastano
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
#el nivell i el diners (la categoria ja no cal)
#segueix una funció polinolica (exponcial)
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values # ha de ser matriu
y = dataset.iloc[:, -1].values

#fem el model linal a veure què pass
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
#transformacio polinomica
poly_reg = PolynomialFeatures(degree = 4) #es busca més ajust (per defecte és 2)
X_poly = poly_reg.fit(X)
X_poly = poly_reg.transform(X)
#X_poly = poly_reg.fit_transform(X) #es el mateix que les dues de dalt

#consctucció del model polinomic a partir de la transformació polinomica
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Polynomial Regression results i el lineal per compara
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#prediccions
lin_reg.predict([[6.5]])
#cal fer el fit transform pels nivells de la funcio polinomica
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))


# Visualising the Polynomial Regression results i el lineal per compara
#fem el grafic amb més un punts 

x_grid = np.arange(1,10, 0.1)
x_grid = x_grid.reshape(len(x_grid),1)

plt.scatter(X, y, color = 'red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.plot(x_grid, lin_reg.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
