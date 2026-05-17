import sympy as sp


def state_to_transfer(A, B, C, D):

    # simbolo
    s = sp.symbols('s')

    # dimensione
    n = len(A)

    # matrici
    A = sp.Matrix(A)
    B = sp.Matrix(B)
    C = sp.Matrix(C)
    D = sp.Matrix(D)

    # identità
    I = sp.eye(n)

    print("Usiamo la formula G(s) = C (sI - A)^(-1) B + D\n")

    print("(sI - A)^(-1) = adj(sI - A) / det(sI - A)")
    print("adj(sI - A) è la matrice aggiunta,")
    print("che si può calcolare come la matrice dei coefficienti di (sI - A)^t")
    
    # (sI - A)
    sI_A = s * I - A
    print("\nsI - A =")
    sp.pprint(sI_A)

    sI_A_t = sI_A.transpose()
    print("\n(sI - A)^t =")
    sp.pprint(sI_A_t)

    adj_sI_A = sI_A_t.cofactor_matrix()
    print("\nadj(sI - A)")
    sp.pprint(adj_sI_A)

    print("\nPossiamo riscrivere la formula come:")
    print("G(s) = [(C * adj(sI - A) * B) / det(sI - A)] + D")

    print("\nSe hai già calcolato gli autovalori di A puoi già scrivere det(sI - A):")
    print("det(sI - A) è il polinomio caratteristico che ha generato tali autovalori")
    print("cioè det(sI - A) = (s - lambda1)*(s - lambda2)*...")
    print(f"\ndet(sI - A) = {sI_A.det()}")

    print("\nPer ridurre il numero di conti, potresti guardare se C e B possiedono 0")

    # CALCOLO FINALE

    # inversa
    sI_A_inv = sI_A.inv()

    # funzione di trasferimento: C (sI - A)^(-1) B + D
    G = C * sI_A_inv * B + D

    # risultato
    G_clean = sp.simplify(G[0])
    G_clean = sp.together(G_clean)
    G_clean = sp.factor(G_clean)

    print("\nG(s) =")
    sp.pprint(G_clean)
    print(f"\nG(s) = {G_clean}")

# ---------------------------------------------------------

if __name__ == "__main__":
   
    # Dato un sistema in forma di stato, è possibile convertirlo nella forma
    # a funzione di trasferimento con la seguente formula:
    #   G(s) = C (sI - A)^(-1) B + D
    # Il codice fa solo questo calcolo

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


    state_to_transfer(A, B, C, D)