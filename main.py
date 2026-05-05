from proprieta_sistema import *
import numpy as np


A = [
    [-7, 0, 0, 0],
    [0, 0, -2, 0],
    [0, 2, 0, 0],
    [0, 0, 0, -8]
]

B = [
    [0],
    [1],
    [0],
    [1]
]

C = [
    [1, 0, 1, 0]
]

stabilita(A)
raggiungibilita(A, B)
osservabilita(A, C)
raggiungibilita_PBH(A, B)
osservabilita_PBH(A, C)