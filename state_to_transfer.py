import sympy as sp

# Dato un sistema in forma di stato, è possibile convertirlo nella forma
# a funzione di trasferimento con la seguente formula:
#   G(s) = C (sI - A)^(-1) B + D
# Il codice fa solo questo calcolo

# Può essere utile per capire la relazione tra autovalori e poli
# (i poli sono tutti e solo gli autovalori raggiungibili e osservabili)

# simbolo
s = sp.symbols('s')

A = [
    [0, 1, 0],
    [0, 0, 1],
    [-1, -2, -3]
]

B = [
    [10],
    [0],
    [0]
]

C = [
    [1, 0, 0]
]

D = [
    [0]
]

# ---------------------------------------------------------

# dimensione
n = len(A)

# matrici
A = sp.Matrix(A)
B = sp.Matrix(B)
C = sp.Matrix(C)
D = sp.Matrix(D)

# identità
I = sp.eye(n)

# (sI - A)
M = s * I - A

# inversa
M_inv = M.inv()

# funzione di trasferimento: C (sI - A)^(-1) B + D
G = C * M_inv * B + D

# risultato
G_clean = sp.simplify(G[0])
G_clean = sp.together(G_clean)
G_clean = sp.factor(G_clean)


print("Espressione:")
print()
print(G_clean)
print()
print("Espressione (in un formato diverso):")
print()
sp.pprint(G_clean)