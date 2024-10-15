import numpy as np
import matplotlib.pyplot as plt
from activations import *
from dibujar import *


class MLP:
    def __init__(self, 
                 layers_dims, 
                 hidden_activation=tanh, 
                 output_activation=logistic):
        #stributtes
        self.L = len(layers_dims) - 1
        self.w = [None] * (self.L + 1)
        self.b = [None] * (self.L + 1)
        self.phi = [None] * (self.L + 1)

        #Inicialize weights and biases
        for l in range(1, self.L +1):
            self.w[l] = -1 + 2*np.random.rand(layers_dims[l], layers_dims[l-1])
            self.b[l] = -1 + 2*np.random.rand(layers_dims[l],1)
            if l == self.L:
                self.phi[l] = output_activation
            else:
                self.phi[l] = hidden_activation

    def predict(self, X):
        A = X.copy()
        for l in range(1, self.L+1):
            Z = self.w[l] @ A + self.b[l]
            A = self.phi[l](Z)
        return A 
    
    def fit(self, X, Y, epochs=500, lr=0.1):
        p = X.shape[1]
        for _ in range(epochs):
            #Iniciaze containers
            A = [None] * (self.L + 1)
            dA = [None] * (self.L + 1)
            lg = [None] * (self.L + 1)

            #Propagation
            A[0] = X.copy()
            for l in range(1, self.L + 1):
                Z = self.w[l] @ A[l-1] + self.b[l]
                A[l], dA[l] = self.phi[l](Z, derivative=True)

            #Backpropagation ----redes de halfig y hilton ----
            for l in range(self.L, 0, -1):
                if l == self.L:
                    lg[l] = - (Y - A[l] * dA[l])

                else:
                    lg[l] = (self.w[l+1].T @ lg[l+1]) * dA[l]

            #Gradient Descent
            for l in range(1, self.L + 1):
                self.w[l] -= (lr/p) * lg[l] @ A[l-1].T
                self.b[l] -= (lr/p) * np.sum(lg[l])

#Ejemplo chingon

X = np.array([[0,0,1,1], [0,1,0,1]])
Y = np.array([[0,1,1,0]])

net = MLP((2,10,1))
net.fit(X,Y)
print(net.predict(X))
MLP_binary_draw(X, Y, net)