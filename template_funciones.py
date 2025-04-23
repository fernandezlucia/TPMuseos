import numpy as np

    # Función que construye la matriz de adyacencia del grafo de museos
    # D matriz de distancias, m cantidad de links por nodo
    # Retorna la matriz de adyacencia como un numpy.
def construye_adyacencia(D,m): 
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # recorriendo las filas, anexamos vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # En realidad, elegimos todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int) # Convertimos a entero
    np.fill_diagonal(A,0) # Borramos diagonal para eliminar autolinks
    return(A)

# matriz es una matriz de NxN
# Retorna la factorización LU a través de una lista con dos matrices L y U de NxN.
# Completar! Have fun
def calculaLU(matriz):
    U = np.copy(matriz)
    L = np.zeros_like(matriz)

    for fila in range(len(matriz)):
        for col in range(fila+1, len(matriz)):
            coeficiente = U[col][fila] / U[fila][fila]
            L[col][fila] = coeficiente
            for k in range(fila, len(matriz)):
                U[col][k] -= coeficiente * U[fila][k]

    for i in range(len(matriz)):
        L[i][i] = 1
    return L, U

    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C

def calcula_matriz_C(A): 
    suma_col = np.sum(A, axis = 1)
    K = np.diag(suma_col)
    n = A.shape[0]
    Kinv = np.zeros((n, n))
    for i in range(n):
       if K[i, i] != 0:
           Kinv[i, i] = 1.0 / K[i, i] # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de A traspuesta
    C = np.transpose(A) @ Kinv  # Calcula C multiplicando Kinv y A

    return C

    
def calcula_pagerank(A,alfa):
    # Función para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # d: coeficientes de damping
    # Retorna: Un vector p con los coeficientes de page rank de cada museo
    C = calcula_matriz_C(A)
    N = ... # Obtenemos el número de museos N a partir de la estructura de la matriz A
    M = ...
    L, U = calculaLU(M) # Calculamos descomposición LU a partir de C y d
    b = ... # Vector de 1s, multiplicado por el coeficiente correspondiente usando d y N.
    Up = scipy.linalg.solve_triangular(L,b,lower=True) # Primera inversión usando L
    p = scipy.linalg.solve_triangular(U,Up) # Segunda inversión usando U
    return p

def calcula_matriz_C_continua(D): 
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C en versión continua
    D = D.copy()
    F = 1/D
    np.fill_diagonal(F,0)
    Kinv = ... # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de F 
    C = ... # Calcula C multiplicando Kinv y F
    return C

def calcula_B(C,cantidad_de_visitas):
    # Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matirz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    B = np.eye(C.shape[0])
    for i in range(cantidad_de_visitas-1):
        # Sumamos las matrices de transición para cada cantidad de pasos
    return B


    





