#Funcion para Normalizar los datos 
def normalize(matrix):
  numRows = matrix.shape[0]
  for i in range(numRows):
    matrixRow = matrix.iloc[i]
    matrix.iloc[i] = (matrixRow - matrixRow.min()) / (matrixRow.max() - matrixRow.min())
  return matrix