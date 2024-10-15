import numpy as np
import pandas as pd

class LogisticNeuron:
  def __init__(self, n_inputs):
    self.w = -1 + 2*np.random.rand(n_inputs)
    self.b = -1 + 2*np.random.rand()

  def predict(self, X):
    Z = np.dot(self.w, X) + self.b
    Y_est = 1 / (1 + np.exp(-Z))
    return Y_est

  def predict_class(self, X, umbral=0.5):
    return 1 * (self.predict(X) > umbral)

  def fit(self, X, Y, epochs=1000, lr=0.1):
    p = X.shape[1]
    for _ in range(epochs):
      Yest = self.predict(X)
      self.w += (lr/p) * np.dot((Y-Yest), X.T).ravel()
      self.b += (lr/p) * np.sum(Y-Yest)

#Funcion para Normalizar los datos 
def normalizeMatrix(matrix):
  numRows = matrix.shape[0]
  for i in range(numRows):
    matrixRow = matrix.iloc[i]
    matrix.iloc[i] = (matrixRow - matrixRow.min()) / (matrixRow.max() - matrixRow.min())
  return matrix

#Lectura de los datos
diabetes = pd.read_csv('./A04/diabetes.csv')
#separar ultima columna en un vector Y (vector de resultados esperados)
outcome = diabetes["Outcome"]
#Crear matriz X con los datos (sin el vector Y)
dataSet = diabetes.drop(["Outcome"], axis=1)
#Transponer la matriz X
matrizXT = dataSet.T

#Normalizamos los datos y 
#Convertimos de DataFrame de pandas a un array de numpy
X = normalizeMatrix(matrizXT).to_numpy()
Y = outcome.to_numpy()

#Ejemplo
neuron = LogisticNeuron(8)
neuron.fit(X,Y)
print('Probabilidad: ', neuron.predict(X))
YEstimate = neuron.predict_class(X)
print('Clase', YEstimate)


#Medir desempeño de la neurona
numAciertos = np.sum(YEstimate == Y)
precision = (numAciertos / Y.shape[0]) * 100
print('Numero de Aciertos: ', numAciertos) 
print('Porcentaje de Aciertos: ', precision, ' %')


#libreria pandas (lr = learning rate)
#paso 1
#Leer datos
#Transponer Matriz x
#separar ultima columna en un vector Y
#paso 2
#Normalizar datos de la misma manera entre 0 y 1
#paso 3
#Entrenar los datos net.fit
#paso 4
#medir el desempeño de la neurona
#contar el numero de asiertos, a cuantos le atino
#comparando los dos vectores, Y_EST - Y
#Sacar el porsentaje de prediccion/asiertos
#METODO DE LAS PARALELAS PARA GRAFICAR EN MAS DIMENCIONES