import numpy as np
from fractions import Fraction

"""
x = 3.6666666666666674
f = Fraction(x).limit_denominator(100)
print(f)        # 11/3
print(f.numerator, f.denominator)  # 11 3
"""


def print_routh_table(routh):
    """
    Stampa la tabella di Routh indentata correttamente
    """
    def format_elem(elem):
        if elem != int(elem):
            return str(Fraction(elem).limit_denominator(100))
        else:
            return str(int(elem))

    # Prima passata: calcola tutte le stringhe e trova la larghezza massima
    formatted = []
    for row in routh:
        row_strs = []
        for i, elem in enumerate(row):
            if i == len(row) - 1 and elem == 0:
                row_strs.append("")
            else:
                row_strs.append(format_elem(elem))
        formatted.append(row_strs)

    col_width = max(len(s) for row in formatted for s in row) + 2

    # Seconda passata: stampa allineato
    k = len(routh) - 1
    for row_strs in formatted:
        print(f"{k} | ", end='')
        k -= 1
        for s in row_strs:
            if s:
                print(s.ljust(col_width), end='')
        print()


def routh_table(G):
    """
    Calcola la tabella di Routh
    """

    P = np.array(G)

    routh = [list(P[np.arange(0, len(P), 2)])]
    routh.append(list(P[np.arange(1, len(P), 2)]),)

    # Poiché le prime due righe della tabella di Routh sono note
    # dobbiamo fare len(G) - 2 passaggi

    for i in range(2, len(G)):
        row1 = routh[i - 2]  # la riga sopra sopra l'attuale
        row2 = routh[i - 1]  # la riga sopra l'attuale

        if len(row2) != len(row1):
            row2.append(0)

        # Creiamo la prossima riga
        new_row = []

        # Quanti elementi si hanno ?
        n = len(row1) - 1 if row1[-1] != 0 else len(row1) - 2
        
        for j in range(1, n + 1):
            M = np.array(
                [[row1[0], row1[j]],
                 [row2[0], row2[j]]]
            )
            # ATTENZIONE row2[0] può essere 0
            if row2[0] == 0:
                print("Attenzione: si ha una divisione per 0.")
                print("L'algoritmo non è in grado di risolverlo.")
                exit(1)
            new_row.append(-np.linalg.det(M) / row2[0])

        routh.append(new_row)

    print_routh_table(routh)


# ===================================================================

# data una G(s) come polinomio
G = [1, 3, 5, 4, 2]
# G = [1, 1, 4, 3, 2, 4, 2]
# G = [1, 0, 1, 2]

routh_table(G)