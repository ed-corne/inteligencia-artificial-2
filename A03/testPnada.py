import pandas as pd
import numpy as np

a = [1, 7, 2]

myvar = pd.Series(a, index = ["x", "y", "z"])

print(myvar)
print(myvar["y"])

print(pd.__version__)

df = pd.read_csv('./A03/DataSet1.csv')

print(df.to_string()) 
print("-------------------------")

x = df['x']
y = df['y']

array_columna1 = np.array(df['x'])

# Imprimir el array para verificar
print(array_columna1.T)