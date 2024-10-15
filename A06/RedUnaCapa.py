import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def linear(z, derivate=False):
    a = z
    if derivate:
        da = np.ones(z.shape)
        return a, da
    return a

def logistic(z, derivate=False): #sigmoid
    a = 1 / (1 + np.exp(-z))
    if derivate:
        da = a * (1 - a) 
        return a, da
    return a

def tanh(z, derivate=False):
  a = np.tanh(z)
  if derivate:
    da = (1 + a) * (1 - a)
    return a, da
  return a

def relu(z, derivate=False):
  a = z * (z >= 0)
  if derivate:
    da = np.array(z >= 0, dtype=float)
    return a, da
  return a

def softmax(z, derivate=False):
    e_z = np.exp(z - np.max(z, axis=0))
    a = e_z / np.sum(e_z, axis=0)
    if derivate:
        da = np.ones(z.shape)
        return a, da
    return a

# red nuronal de una capa

class OLN: #ONE LAYER NETWORK
    def __init__(self, n_inputs, n_outputs, activation_function=linear):
        self.w = -1 + 2*np.random.rand(n_outputs,  n_inputs)
        self.b = -1 + 2*np.random.rand(n_outputs, 1)
        self.f = activation_function
    
    def predict(self, X):
      Z = np.dot(self.w, X) + self.b
      Yest = self.f(Z)
      return Yest
    
    def fit(self, X, Y, epochs=1000, lr=0.5):
        p = X.shape[1]
        for _ in range(epochs):
            #propagacion
            Z = self.w @ X + self.b
            Yest, dY, = self.f(Z, derivate=True)

            #gradiente local
            lg = (Y - Yest) * dY

            #Actualizacion de parametros
            self.w += (lr/p) * np.dot(lg, X.T)
            self.b += (lr/p) * np.reshape(np.sum(lg, axis=1), (-1, 1))

#Lectura de los datos
dataset = pd.read_csv('./A06/Dataset_A05.csv')
X = dataset[["x1","x2"]]
Y = dataset[["y1","y2","y3","y4"]]
#Transponer matrices
Y = Y.T.to_numpy()
X = X.T.to_numpy()

#Dibujo
def plot_data(X,Y,net):
  dot_c = ('red', 'green', 'blue', 'black')
  lin_c = ('r-', 'g-', 'b-', 'k-')
  for i in range(X.shape[1]):
    c = np.argmax(Y[:,i])
    plt.scatter(X[0,i], X[1,i], color = dot_c[c], edgecolors='k')

  for i in range(4):
    w1, w2, b = net.w[i,0], net.w[i,1], net.b[i]
    plt.plot([-0,1], [(-b/w2), (1/w2)*(-w1-b)], lin_c[i])

  plt.xlim([0,1])
  plt.ylim([0,1])
#Ejemplo
net = OLN(2, 4, logistic)
net.fit(X, Y,  epochs=2000)
plot_data(X, Y, net)
plt.show()



