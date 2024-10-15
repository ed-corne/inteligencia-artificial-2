'''
2.1 Crea datos aleatorios de personas (peso y altura) y decide si tienen sobre peso usando la formula del IMC.
2.2 Normaliza los datos (con el método de tu preferencia).
2.3 Entrena el perceptrón.
2.4 Prueba con nuevos datos (recuerda que para meterlos al perceptrón deben de ser normalizados antes)
'''
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
    plt.plot([-2,2],[(1/w2)*(-w1*(-2)-b), (1/w2)*(-w1*(2)-b)], 'g')

#DATOS

def generar_datos_normalizados(num_personas):
    # Generar datos aleatorios de peso (en kg) y altura (en metros)
    pesos = np.random.uniform(50, 100, num_personas)  # Pesos entre 50 kg y 100 kg
    alturas = np.random.uniform(1.5, 2.0, num_personas)  # Alturas entre 1.5 m y 2.0 m

    # Calcular IMC
    imcs = pesos / (alturas ** 2)

    # Determinar si tiene sobrepeso (IMC >= 25)
    sobrepeso = (imcs >= 25).astype(int)

    # Normalizar los pesos y alturas entre 0 y 1
    pesos_norm = (pesos - pesos.min()) / (pesos.max() - pesos.min())
    alturas_norm = (alturas - alturas.min()) / (alturas.max() - alturas.min())

    # Crear la matriz X con los pesos y alturas normalizados
    X = np.array([pesos_norm, alturas_norm])

    # Crear el vector Y con 1 si tiene sobrepeso, 0 si no
    Y = sobrepeso

    return X, Y


X, Y = generar_datos_normalizados(100)

model = Neurona_binaria(2, 0.5)

model.fit(X, Y)
print(model.predict(X))

#Dibujo
#Usa indexación booleana en lugar de un bucle for
# Y == 0 dará [False, True, False, True]
plt.plot(X[0, Y == 0], X[1, Y == 0], 'or', label='No sobrepeso')  # Puntos rojos
plt.plot(X[0, Y == 1], X[1, Y == 1], 'ob', label='Sobrepeso')     # Puntos azules


draw_2d_neuron(model)
plt.title('Perseptron - Clasificación de Sobrepeso')
plt.xlabel('Peso Normalizado')
plt.ylabel('Altura Normalizada')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.grid(True)
plt.legend(loc='upper right')  # Coloca la leyenda en la esquina superior derecha

plt.show()