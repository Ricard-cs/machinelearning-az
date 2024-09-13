# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 10:47:29 2024

@author: rcastano
"""

import cx_Oracle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#cx_Oracle.init_oracle_client(lib_dir=r"C:\instantclient_23_5")

# Conexión a la base de datos
dsn = cx_Oracle.makedsn("teo", "1527", service_name="fdt")
conexion = cx_Oracle.connect(user="system", password="sapadm99", dsn=dsn)

# Crear un cursor para ejecutar consultas
cursor = conexion.cursor()

# Ejecutar una consulta SQL (por ejemplo, selecciona todos los registros de una tabla)
query = """select a.matnr, c.VKORG, e.BZIRK, e.INCO1, BRSCH, d.land1, d.regio, FKIMG, a.netwr/FKIMG 
from sapsr3.VBRP  a inner join sapsr3.VBRK  b on a.vbeln = b.vbeln
inner join sapsr3.vbak c on c.vbeln = a.aubel
inner join sapsr3.kna1 d on d.kunnr = c.kunnr 
inner join sapsr3.VBKD e on e.vbeln = c.vbeln
where a.erdat > '20200000' and 
(b.FKART = 'F2' OR b.FKART = 'F2FR') and 
b.waerk = 'EUR' 
and a.netwr > 0 AND 
MATNR <> 'PORTES' AND 
MATNR NOT LIKE  'INSTAL%'  AND 
MATNR NOT LIKE '%PE' AND 
MATNR NOT IN ('DTOTR','DTOMU','DTOJD','DTOAP') where matnr = PA600"""

cursor.execute(query)

columnas = [col[0] for col in cursor.description]

# Convertir el resultado de la consulta en un DataFrame de pandas
df = pd.DataFrame(cursor.fetchall(), columns=columnas)

print (df.head())
# Cerrar el cursor y la conexión
cursor.close()
conexion.close()