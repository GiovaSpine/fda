from proprieta_sistema import *
import numpy as np


A = [
    [-7, 0, 0, 0],
    [0, 0, -2, 0],
    [0, 2, 0, 0],
    [0, 0, 0, -8]
]

print(np.matrix(A), "\n")

stabilita(A)

B = [
    [0],
    [1],
    [0],
    [1]
]

raggiungibilita(A, B)

C = [
    [1, 0, 1, 0]
]

osservabilita(A, C)



A = [
    [-1, 1, 0, 0, 0, 0, 0],
    [0, -1, 1, 0, 0, 0, 0],
    [0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, -1, 1, 0, 0],
    [0, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0, 2, 1],
    [0, 0, 0, 0, 0, 0, 2]
]

B = [
    [0, 0],
    [0, 0],
    [0, 1],
    [0, 0],
    [1, 0],
    [0, 0],
    [1, 1]
]

B = [
    [0],
    [0],
    [1],
    [0],
    [1],
    [0],
    [1]
]

raggiungibilita2(A, B)
osservabilita2(A, C)
