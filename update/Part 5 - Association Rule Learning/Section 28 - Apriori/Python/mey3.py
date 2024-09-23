# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:37:34 2024

@author: rcastano
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Cargar las transacciones desde un archivo CSV (o tu propio DataFrame)
# Reemplaza 'transacciones.csv' con el nombre del archivo que tengas
transacciones_df = pd.read_csv('arts_2023.csv')

# Suponiendo que cada fila es una transacción con los ítems separados por comas
# Convertir los ítems en formato de transacción binaria (One-Hot Encoding)
transacciones_onehot = transacciones_df.groupby('transaction_id')['item'].apply(lambda x: x.str.join('|').str.get_dummies())

# Filtrar los 100 ítems más vendidos (aquellos con más de 11187 apariciones)
item_frecuencia = transacciones_onehot.sum().sort_values(ascending=False)
items_mas_vendidos = item_frecuencia[item_frecuencia >= 11187].head(100)

# Filtrar las transacciones solo con los 100 ítems más vendidos
transacciones_filtradas = transacciones_onehot[items_mas_vendidos.index]

# Aplicar el algoritmo Apriori
frequent_itemsets = apriori(transacciones_filtradas, min_support=0.05, use_colnames=True)

# Generar reglas de asociación a partir de los conjuntos frecuentes
reglas = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Mostrar las reglas encontradas
print(reglas)
