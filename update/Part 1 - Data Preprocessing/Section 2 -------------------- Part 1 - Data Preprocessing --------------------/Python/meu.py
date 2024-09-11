# Data Preprocessing Template

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#treure NAN (missing)
#categoritzar
#conjunt d'entreno
#escalar (estandarització / normalització) 


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

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#escalat (a vegades cal o no cal)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit(X_test)
#Y no es fa ja que es de classificació