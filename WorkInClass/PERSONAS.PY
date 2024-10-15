import numpy as np

def generar_datos(num_personas):
    # Generar datos aleatorios de peso (en kg) y altura (en metros)
    pesos = np.random.uniform(50, 100, num_personas)  # Pesos entre 50 kg y 100 kg
    alturas = np.random.uniform(1.5, 2.0, num_personas)  # Alturas entre 1.5 m y 2.0 m

    # Calcular IMC
    imcs = pesos / (alturas ** 2)

    # Determinar si tiene sobrepeso (IMC >= 25)
    sobrepeso = (imcs >= 25).astype(int)

    # Crear la matriz X con pesos y alturas
    X = np.array([pesos, alturas])

    # Crear el vector Y con 1 si tiene sobrepeso, 0 si no
    Y = sobrepeso

    return X, Y

# Ejemplo de uso
num_personas = 4  # Puedes cambiar este valor para generar más o menos datos
X, Y = generar_datos(num_personas)

print("X:\n", X)
print("Y:\n", Y)




#normalizados

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

