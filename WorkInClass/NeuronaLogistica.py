import numpy as np

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

  def fit(self, X, Y, epochs=500, lr=0.1):
    p = X.shape[1]
    for _ in range(epochs):
      Yest = self.predict(X)
      self.w += (lr/p) * np.dot((Y-Yest), X.T).ravel()
      self.b += (lr/p) * np.sum(Y-Yest)


# Ejemplo

X = np.array([[0,1,0,1],
              [0,0,1,1]])

Y = np.array([0,0,0,1])

neuron = LogisticNeuron(2)
neuron.fit(X,Y)
print('Probabilidad: ', neuron.predict(X))
print('Clase', neuron.predict_class(X))

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