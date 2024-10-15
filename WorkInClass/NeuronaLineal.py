import numpy as np
import matplotlib.pyplot as plt

class Neurona_binaria:
    def __init__(self, n_inputs):
        #atributo de la clase
        #aleatorio, uniformente distribuido entre -1 y 1
        self.w = -1 + 2*np.random.rand(n_inputs)
        self.b = -1 + 2*np.random.rand()

    def predict(self, X):
        Yest = np.dot(self.w, X) + self.b
        return Yest
    
    # Version media entre BGD y SGD
    #miniBatch (el mas usado)
    def batcher(self, X, Y, size):
        p = X.shape[1] #valor de columnas
        li, lu = 0, size #limite inferior limite superior
        while True:
            if li < p:
                yield X[:, li:lu], Y[:, li:lu] #funcion generadora
                li, lu = + size, lu + size
        else:
            return None

    def fit(self, X, Y, epochs=50, lr=0.1, solver='BGD'):
      p = X.shape[1]

      #SGD
      if solver == 'SGD':
         for _ in range(epochs):
            for i in range(p):
               yest = self.predict(X[:,i])
               self.w += lr*(Y[:,i]-yest)*X[:,i]
               self.b += lr*(Y[:,i]-yest)
      
      #BGD
      elif solver == 'BGD':
         for _ in range(epochs):
            Yest = self.predict(X)
            self.w += (lr/p) * ((Y-Yest) @ X.T).ravel()
            self.b += (lr/p) * np.sum(Y-Yest)

      #Pseudoinverse
      else:
        #crear nueva matriz X
        Xhat = np.concatenate((np.ones(1,p),X), axis=0)
        #calcular w
        what = np.dot(Y.reshape(1,-1), np.linalg.pinv(Xhat))
        self.b = what[0,0]
        self.w = what[0,1:]


#Test code

p = 100
x = -1 + 2*np.random.rand(1,p)
y = -18*x + 6 + 3 * np.random.randn(1,p)

neuron = Neurona_binaria(1)
neuron.fit(x, y, solver='BGD', lr=1)

#Dibujo 
plt.plot(x, y, 'b. ')
xn = np.array([[-1,1]])
plt.title('Neurona Lineal - SGD')
plt.plot(xn.ravel(), neuron.predict(xn), '--r')
plt.show()
