import numpy as np
import sympy as sp
from fractions import Fraction


def print_routh_table(routh, additional_conditions=[]):
    """
    Stampa la tabella di Routh indentata correttamente
    """
    def format_elem(elem):
        if isinstance(elem, float):
            return str(Fraction(elem).limit_denominator(100))
        elif isinstance(elem, int):
            return str(int(elem))
        else:
            return str(elem)
            

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
    print()
    
    # Se si aveano condizioni aggiuntive, vanno stampate
    print("Condizioni aggiuntive:")
    for ac in additional_conditions:
        print("{ ", ac, " > 0", sep='')


def routh_table(G, simplify=False):
    """
    Calcola la tabella di Routh

    Parametri:
        G (list): La funzione di trasferimento
        simplify (bool): Indica se è richiesto (True) la semplificazione
                         dell'espressione, moltiplicando ogni riga per
                         il denominatore del primo elemento.
    """

    P = np.array(G)

    routh = [list(P[np.arange(0, len(P), 2)])]
    routh.append(list(P[np.arange(1, len(P), 2)]),)

    # Poiché le prime due righe della tabella di Routh sono note
    # dobbiamo fare len(G) - 2 passaggi

    # Condizioni aggiuntive per valutare la stabilità
    additional_conditions = []

    for i in range(2, len(G)):
        row1 = routh[i - 2]  # La riga sopra sopra l'attuale
        row2 = routh[i - 1]  # La riga sopra l'attuale

        if len(row2) != len(row1):
            # Aggiungiamo uno 0 in fondo se neccessario
            row2.append(0)

        # Creiamo la prossima riga
        new_row = []

        # Quanti elementi si hanno ?
        n = len(row1) - 1 if row1[-1] != 0 else len(row1) - 2
        
        for j in range(1, n + 1):
            # Il nuovo elemento è meno il determinante della matrice 2x2 che si ottiene
            # dalla colonna a sinistra con la colonna attuale, diviso row2[0], ma:
            # ATTENZIONE row2[0] può essere 0
            if row2[0] == 0:
                # consideriamo 0 come E (epsilon)
                row2[0] = sp.Symbol('E')

            new_elem = (row2[0] * row1[j] - row1[0] * row2[j]) / row2[0]

            if simplify:
                if j == 1:
                    # Se si ha un denominatore con simboli per il primo elemento,
                    # moltiplichiamolo nella riga per semplificarla
                    num, den = sp.fraction(new_elem)
                    if not den.free_symbols:
                        # den è una costante numerica
                        den = 1.0
                    else:
                        # den contiene simboli, lo useremo per moltiplicarlo con la riga
                        # LO DOBBIAMO SEGNARE, sennò si perde tale condizione
                        additional_conditions.append(den)
                        new_elem = num
                        new_elem = sp.expand(new_elem)  # espandi
                else:
                    new_elem *= den
            
            new_row.append(new_elem)

        routh.append(new_row)
    
    # Attenzione: se abbiamo moltiplicato i denominatori per le righe
    # li dobbiamo ricordare per la valutazione della stabilità

    if simplify:
        print_routh_table(routh, additional_conditions=additional_conditions)
    else:
        print_routh_table(routh)


# ===================================================================

# data una G(s) come polinomio
# es. [1, 3, 5, 4, 2] --> s^4 + 3s^3 + 5s^2 + 4s + 2
# si possono usare simboli, definendoli con sp.Symbol('k')
# es.
#   k = sp.Symbol('k')
#   G = [1, 3, 2, k] --> s^3 + 3s^2 + 2s + k

G = [1, 3, 5, 4, 2]
G = [1, 1, 4, 3, 2, 4, 2]
#G = [1, 0, 1, 2]

k = sp.Symbol('k')
G = [1, 3, 2, k]

b = sp.Symbol('b')
c = sp.Symbol('c')
G = [1, b, c, 1]

G = [1, 20-k, 7, 5, 1]

routh_table(G, simplify=True)