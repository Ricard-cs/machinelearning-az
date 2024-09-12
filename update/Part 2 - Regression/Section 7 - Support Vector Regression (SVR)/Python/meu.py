# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 12:59:46 2024

@author: rcastano
"""

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

#no dividim entre test i train perquè hi ha molt poques dades

#important SVM cal fer escalat!!! sinó no funciona bé

#dos escaladors diferents (per x i per y)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X) #passem array
y = y.reshape(len(y),1) #passem array
y = sc_y.fit_transform(y)
print(X)
print(y)


# model
from sklearn.svm import SVR
#transformacio polinomica
svr_reg = SVR(kernel= "rbf")
y =y.ravel() #passem a vector de nou
X_svr = svr_reg.fit(X,y)

#al estar escalat cal enviar el 6.5 escalat 
sc_y.inverse_transform(X_svr.predict(sc_X.transform([[6.5],[7]])).reshape(-1,1))

#desescalem i tenim el valor
x_desescalat = sc_X.inverse_transform(X)
y_desescalat = sc_y.inverse_transform(y.reshape(-1,1))

x_grid = np.arange(min(x_desescalat),max(x_desescalat), 0.1).reshape(-1,1)
y_predit_desescalat = sc_y.inverse_transform(X_svr.predict(sc_X.transform(x_grid)).reshape(-1,1))


#fem el grafic amb més un punts per suabitzar (sinó substituir per X)
plt.scatter(x_desescalat, y_desescalat, color = 'red')
plt.plot(x_desescalat, y_desescalat, color = 'blue')
plt.plot(x_grid, y_predit_desescalat, color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

