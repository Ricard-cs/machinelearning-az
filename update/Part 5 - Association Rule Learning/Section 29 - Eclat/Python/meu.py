# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:45:50 2024

@author: rcastano
"""

# Eclat

# Run the following command in the terminal to install the apyori package: pip install apyori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('arts_2023.csv')
transactions = []
for i in range(0, 25705):
  transactions.append([str(dataset.values[i,j]) for j in range(1, 137)])
print("1")
# Training the Eclat model on the dataset
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.04, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

# Visualising the results

## Displaying the first results coming directly from the output of the apriori function
results = list(rules)
results

## Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support'])

## Displaying the results sorted by descending supports
resultsinDataFrame.nlargest(n = 10, columns = 'Support')