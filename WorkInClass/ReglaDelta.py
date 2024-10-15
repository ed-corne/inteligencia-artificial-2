import numpy as np

def linear(z, derivative=True):
    a = z
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a

def tanh(z, derivative=True):
    a = 1/(1+np.exp(-z))
    if derivative:
        da = (1+a)*(1-a)
        return a, da
    return a

def linear(z, derivative=True):
    a = z
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a


class neuron:
    def __init__(self, n_inputs, activation_function=linear):
        self.w = -1 + 2*np.random.rand(n_inputs)
        self.b = -1 + 2*np.random.rand()
        self.f = activation_function #create an alias for the activation function
    #propagacion
    def predict(self, X):
        z = np.dot(self.w, X) + self.b
        return self.f(z)
    
    def fit(self, x, y, epochs=500, lr=1):
        p = x.shape[1]


        #propagar neurona
        z = np.dot(self.w, x) + self.b
        y_est, dy = self.f(z, derivate=True)

        #gradiente local
        lg = (y-y_est) * dy

        #actualizar parametros
        self.w += (lr/p) * np.dot(lg, x.T).ravl()
        self.b += (lr/p) * np.sum(lg)



