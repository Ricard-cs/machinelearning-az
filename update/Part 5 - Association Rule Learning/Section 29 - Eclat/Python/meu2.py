# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 13:42:04 2024

@author: rcastano
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Paso 3: Crear el DataFrame de transacciones
transactions = [
    {'milk', 'bread', 'butter'},
    {'bread', 'butter'},
    {'milk', 'butter'},
    {'bread'},
    {'milk', 'bread', 'butter'},
    {'butter'}
]

# Convertir las transacciones a formato One-Hot Encoding
all_items = sorted(set(item for transaction in transactions for item in transaction))
encoded_data = []

for transaction in transactions:
    encoded_data.append([1 if item in transaction else 0 for item in all_items])

df = pd.DataFrame(encoded_data, columns=all_items)

# Paso 4: Usar apriori para generar itemsets frecuentes con min_support
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

# Mostrar los itemsets frecuentes
print(frequent_itemsets)

# Paso 5: Generar las reglas de asociación
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.3)

# Mostrar las reglas de asociación
print(rules)