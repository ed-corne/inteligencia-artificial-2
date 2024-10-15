import numpy as np

def linear(z, derivate=False):
    a = z
    if derivate:
        da = np.ones(z.shape)
        return a, da
    return a

def logistic(z, derivate=False):
    a = 1 / (1 + np.exp(-z))
    if derivate:
        da = np.ones(z.shape)
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
        Z = self.w @ X + self.b
        return self.f(Z)
    
    def fit(self, X, Y, epochs=1000, lr=0.1):
        p = X.shape[1]
        for _ in range(epochs):
            #propagacion
            Z = self.w @ X + self.b
            Yest, dY, = self.f(Z, derivate=True)

            #gradiente local
            lg = (Y - Yest) * dY

            #Actualizacion de parametros
            self.w += (lr/p) * lg @ X.T
            self.b += (lr/p) * np.sum(lg)


