"""
Compuerta Logica AND, OR, XOR
"""
import numpy as np
import matplotlib.pyplot as plt

class Neurona_binaria:
    def __init__(self, n_inputs, learning_rate):
        #atributo de la clase
        #aleatorio, uniformente distribuido entre -1 y 1
        self.w = -1 + 2*np.random.rand(n_inputs)
        self.b = -1 + 2*np.random.rand()
        self.eta = learning_rate

    def predict(self, X):
        #cuantas ... tiene
        p = X.shape[1]
        y_est = np.zeros(p)
        for i in range(p):   
            #Se asegura de que se calcula como un escalar (no un array)
            y_est_i = np.dot(self.w, X[:,i]) + self.b
            y_est[i] = 1 if y_est_i >= 0 else 0   
            #puede devolver un array en lugar de un escalar en ciertas circunstancias.
            #y_est[i] = np.dot(self.w, X[:,i]) + self.b
            #if y_est[i] >= 0:
                #y_est[i]=1
            #else:
                #y_est[i]=0
        return y_est

    def fit(self, X, Y, epochs=50):
        p = X.shape[1]
        for _ in range(epochs):
            for i in range(p):
                y_est = self.predict(X[:,i].reshape(-1,1))
                self.w += self.eta * (Y[i] - y_est) * X[:,i]
                self.b += self.eta * (Y[i] - y_est)

#Testing code....................

def draw_2d_neuron(model):
    w1, w2, b = model.w[0], model.w[1], model.b
    plt.plot([-2,2],[(1/w2)*(-w1*(-2)-b), (1/w2)*(-w1*(2)-b)])

#DATOS

X = np.array([[0,0,1,1],
              [0,1,0,1]])
Y = np.array( [1,1,1,0]) #Resultado esperado

model = Neurona_binaria(2, 0.5)

model.fit(X, Y)
print(model.predict(X))

#Dibujo

p = X.shape[1]
for i in range(p):
    if Y[i] == 0:
      plt.plot(X[0,i], X[1, i], 'or')
    else:
      plt.plot(X[0,i], X[1,i], 'ob')

draw_2d_neuron(model)
plt.title('Perseptron')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.xlabel('r$x_1$')
plt.ylabel('r$x_2$')
plt.show()