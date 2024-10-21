import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from activations import *
from dibujar import *
from normalize import *

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

#---------- Ejemplo XOR -------------------
X = np.array([[0,0,1,1], [0,1,0,1]])
Y = np.array([[0,1,1,0]])

net = MLP((2,10,1))
net.fit(X,Y)
print(net.predict(X))
MLP_binary_draw(X, Y, net)

#--------------- Ejemplo Moons ----------------------------
#Lectura de los datos
moons = pd.read_csv('./WorkInClass/Multicapa/moons.csv')
#separar ultima columna en un vector Y (vector de resultados esperados)
Y_moons = moons["y"]
#Crear matriz X con los datos (sin el vector Y)
X_moons = moons.drop(["y"], axis=1)
#Transponer la matriz X
X_moons = X_moons.T.to_numpy()
Y_moons = Y_moons.T.to_numpy()

net = MLP((2,50,30,1))
net.fit(X_moons,Y_moons)
print(net.predict(X_moons))
MLP_binary_draw2(X_moons,Y_moons, net)

#--------- Ejempli Blobs ----------------
blobs = pd.read_csv('./WorkInClass/Multicapa/blobs.csv')
Y_blobs = blobs["y"]
X_blobs = blobs.drop(["y"], axis=1)
X_blobs = X_blobs.T.to_numpy()
Y_blobs = Y_blobs.T.to_numpy()

net = MLP((2,40,20,1))
net.fit(X_blobs,Y_blobs)
print(net.predict(X_blobs))
MLP_binary_draw3(X_blobs,Y_blobs, net)

#-------- Ejemplo Circles ----------------
circles = pd.read_csv('./WorkInClass/Multicapa/circles.csv')
Y_circles = circles["y"]
X_circles = circles.drop(["y"], axis=1)
X_circles = X_circles.T.to_numpy()
Y_circles = Y_circles.T.to_numpy()

net = MLP((2,70,40,1))
net.fit(X_circles,Y_circles)
print(net.predict(X_circles))
MLP_binary_draw4(X_circles,Y_circles, net)
