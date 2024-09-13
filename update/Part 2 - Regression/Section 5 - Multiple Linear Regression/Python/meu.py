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

#ell no ho fa, però sembla que els valor 0 no son correctes
# from sklearn.impute import SimpleImputer
# SimpleImputer = SimpleImputer(missing_values = 0, strategy = "mean")
# SimpleImputer.fit(X[:,0:3])
# X[:, 0:3] = SimpleImputer.transform(X[:, 0:3])


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


#contruccio model opitm amb eliminacio cap errere
import statsmodels.api as sm
#afegim una columna d'1 en les X perquè cal veure si no només els coeficients sinó el valor independent (intercept) és el que cal eliminar
#X = np.append(arr=X, values=np.ones((50,1)).astype(int), axis = 1)
#posar els 1 primer tipus int
X = np.append(arr=np.ones((50,1)).astype(int), values=X, axis = 1)

#si volem afegir una fila (no fer és per saber-ho)
#X = np.append(arr=np.ones((1,6)).astype(int), values=X, axis = 0)


#variable amb les variables importants


#definim el nivell de significació que servirà per determinar si la variable ha d'estar en el model o no (normalment 0.05)
SL = 0.05


x_opt = X[:, [0, 1, 2, 3, 4, 5]]
#cal posar una columna plena d'uns per tenir en compte la intercept (si cal eliminar-la)
regresio_OLS = sm.OLS(endog =y, exog = x_opt.tolist()).fit()
regresio_OLS.summary()


#quant més baix és p>|t| (p-valor) més important es la variable (const i x3 son més significatives, la resta no perqué son majors que 0.05)

#x2 és la que té el valor més alt i superior a 0.05 (varibale dummy nova york) i per tal cal eliminar-la

x_opt = X[:,[0,1,3,4,5]]
#cal posar una columna plena d'uns per tenir en compte la intercept (si cal eliminar-la)
regresio_OLS = sm.OLS(endog =y, exog = x_opt.tolist()).fit()

regresio_OLS.summary()

x_opt = X[:,[0,3,4,5]]
#cal posar una columna plena d'uns per tenir en compte la intercept (si cal eliminar-la)
regresio_OLS = sm.OLS(endog =y, exog = x_opt.tolist()).fit()

regresio_OLS.summary()

x_opt = X[:,[0,3,5]]
#cal posar una columna plena d'uns per tenir en compte la intercept (si cal eliminar-la)
regresio_OLS = sm.OLS(endog =y, exog = x_opt.tolist()).fit()
regresio_OLS.summary()

x_opt = X[:,[0,3]]
#cal posar una columna plena d'uns per tenir en compte la intercept (si cal eliminar-la)
regresio_OLS = sm.OLS(endog =y, exog = x_opt.tolist()).fit()

regresio_OLS.summary()

print(regressor.coef_.min())
print(regressor.intercept_)

#automatizació de la modelització

