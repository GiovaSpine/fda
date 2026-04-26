import numpy as np
import sympy as sp


def spectral_info(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]

    # autovalori numerici
    eigvals = np.linalg.eigvals(A)

    # raggruppo autovalori "uguali" (tolleranza numerica)
    eigvals_rounded = np.round(eigvals, decimals=8)

    unique_vals = {}
    for lam in eigvals_rounded:
        unique_vals[lam] = unique_vals.get(lam, 0) + 1

    result = []

    for lam, alg_mult in unique_vals.items():

        # molteplicità geometrica = dim ker(A - λI)
        M = A - lam * np.eye(n)

        rank = np.linalg.matrix_rank(M)
        geom_mult = n - rank

        result.append({
            "autovalore": lam,
            "molteplicità_algebrica": alg_mult,
            "molteplicità_geometrica": geom_mult
        })

    for r in result:
        print(r)


def jordan_form(A):
    """
    Restituisce forma di Jordan, matrice di trasformazione P e J tali che A = P J P^{-1}
    """
    A = sp.Matrix(A)

    P, J = A.jordan_form()

    result =  {
        "J (Jordan form)": J,
        "P (cambio di base)": P,
        "P_inv": P.inv()
    }

    for key, value in result.items():
        print(f"{key}:\n{np.array(value)}\n")


A = [
    [1, 2, 1, 0],
    [-2, 1, 0, 1],
    [0, 0, 1, 2],
    [0, 0, -2, 1]
]

# su wolframalpha
# jordan form {{1,2,1,0},{-2,1,0,1},{0,0,1,2},{0,0,-2,1}}

spectral_info(A)
jordan_form(A)
