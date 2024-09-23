# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:19:05 2024

@author: rcastano
"""

# Apriori

# Run the following command in the terminal to install the apyori package: pip install apyori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing


dataset = pd.read_csv('arts_2023.csv', usecols=range(1, 137), dtype=str)

# Reemplazar los valores NaN por la cadena 'nan'
dataset.fillna('nan', inplace=True)

# Convertir a lista de listas
transactions = dataset.values[:25705].tolist()



# Training the Apriori model on the dataset
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.0005, min_confidence = 0.02, min_lift = 3, min_length = 1)
#mirar el video per veure els hiperparams

# Visualising the results

## Displaying the first results coming directly from the output of the apriori function
results = list(rules)
results

## Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

## Displaying the results non sorted
resultsinDataFrame

## Displaying the results sorted by descending lifts
resultsinDataFrame.nlargest(n = 100, columns = 'Lift')

