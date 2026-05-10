import control as ctrl
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from utily import *


def quadranti(min_phase, max_phase):
    """
    Stampa in quali quadranti si trova il diagramma di Nyquist in base alla fase.
    Si ricoda che:
    
        secondo | primo
        --------|--------
         terzo  | quarto
    """
    
    def normalizza_fase(phase):
        # porta in [0, 360)
        a = phase % 360 
        return a
    
    min_phase = normalizza_fase(min_phase)
    max_phase = normalizza_fase(max_phase)

    quadrante = ""

    if max_phase <= 90:
        quadrante = "primo"
    elif max_phase <= 180.0:
        if min_phase >= 90.0:
            quadrante = "secondo"
        else:
            quadrante = "primo e secondo"
    elif max_phase <= 270.0:
        if min_phase >= 180.0:
            quadrante = "terzo"
        elif min_phase >= 90.0:
            quadrante = "secondo e terzo"
        else:
            quadrante = "primo, secondo e terzo"
    else:
        if min_phase >= 270.0:
            quadrante = "quarto"
        elif min_phase >= 180.0:
            quadrante = "terzo e quarto"
        elif min_phase >= 90.0:
            quadrante = "secondo, terzo e quarto"
        else:
            quadrante = "primo, secondo, terzo e quarto"

    print(f"Siamo nei quadranti: {quadrante}")
    print()


def calcola_G_jw(num, den):
    """
    Data G(s) come num e den, calcola G(jw).
    """

    s, w = sp.symbols('s w', real=True)
    j = sp.I

    # Costruisce G(s) come polinomio simbolico
    def lista_a_polinomio(coeffs, var):
        grado = len(coeffs) - 1
        return sum(c * var**(grado - i) for i, c in enumerate(coeffs))

    num_poly = lista_a_polinomio(num, s)
    den_poly = lista_a_polinomio(den, s)
    
    G_s = num_poly / den_poly
    print(f"G(s)   = {G_s}")

    # Sostituisce s = jw
    G_jw = G_s.subs(s, j*w)
    G_jw = sp.simplify(G_jw)
    print(f"G(jw)  = {G_jw}")

    # Separa in a + jb
    re = sp.simplify(sp.re(G_jw))
    im = sp.simplify(sp.im(G_jw))
    print(f"G(jw) = a(w) + j*b(w) = ({re}) + j({im})")
    print()

    return G_jw


def analisi_limiti(G_jw):
    """
    Data G(jw) calcola i limiti per w->0 e w->inf.
    
    Parametri:
        G_jw: La risposta in frequenza
    """
    
    w = sp.symbols('s w', real=True)

    n = len(den) - 1  # grado numeratore
    m = len(num) - 1  # grado denominatore

    # Limite w -> 0
    # Si comporta circa come a(jw)^k1 / b(jw)^k2, dove k1 e k2 sono gradi minimi
    print(f"Limite w --> 0 = ", end='')
    lim0 = sp.limit(G_jw, w, 0)
    lim0 = sp.simplify(lim0)
    print(f"G(j*0) = {lim0}")
    # Per il grado minimo dobbiamo considerare:
    # l'indice dei numeri più a destra nelle liste num e den, diversi da 0
    k1, a = next((i, c) for i, c in enumerate(reversed(num)) if c != 0)
    k2, b = next((i, c) for i, c in enumerate(reversed(den)) if c != 0)
    if k1 == k2:
        # Tende ad un numero finito a/b
        print(f"La funzione tende ad un numero finito {a}/{b} = {a / b}")
        print(f"La fase: {0 if (a / b) >= 0 else -180.0}")
    elif k1 < k2:
        # Tende all'infinito
        print(f"La funzione tende all'infinito come 1/(jw)^{k2 - k1}")
        print(f"La fase: {(k2 - k1) * -90.0}")
    else:
        # Tende a 0
        print(f"La funzione tende a 0 come (jw)^{k1 - k2}")
        print(f"La fase: ?")


    # Limite w -> inf
    # Si comporta circa come 1 / (jw)^(n-m)
    # (allora l'ampiezza è sempre zero ? dipende se m < n)
    print(f"\nLimite w --> inf = ", end='')
    lim_inf = sp.limit(G_jw, w, sp.oo)
    lim_inf = sp.simplify(lim_inf)
    print(f"G(j*inf) = {lim_inf}")
    if m < n:
        # Tende a 0
        print(f"La funzione tende a 0 come 1/(jw)^{n - m}")
        print(f"La fase: {(n - m) * -90.0}")
    elif m == n:
        # Tende ad un numero finito a/b
        print(f"La funzione tende ad un numero finito {num[0]}/{den[0]} = {num[0] / den[0]}")
        print(f"La fase: {0 if (num[0] / den[0]) >= 0 else -180.0}")
    else:
        # IMPOSSIBILE per i sistemi causali
        raise ValueError(f"ERRORE: il grado del numeratore {m} è maggiore di quello del denominatore {n}")

    return G_jw, lim0, lim_inf


# ===================================================================

# P(s) = N(s) / D(s)
# Dove N(s) e D(s) sono polinomi qualsiasi (non per forza monico)

# es. 2s^2 + 3 diventa [2, 0, 3]

num = [1, 2, 1]
den = [1, 200, 10000, 0]

num = [1, 20, 100]
den = [1, 200, 10000, 0]

num = [1]
den = [1, 1,0, 0]

num = [3.6]
den = [1, 4, 9, 0]


G = ctrl.TransferFunction(num, den)


# Range di frequenze
# es. np.logspace(-2, 3, 1000) va da 10^-2 a 10^3
# può essere necessario ingrandirlo, perché serve per trovare il range della fase
# (si suggerisce di fare i diagrammi di Bode e guardare da quelli il range di frequenze, se necessario)
LEN = 5000
omega = np.logspace(-10, 10, LEN)


# ---------------------------------

print(G, "\n")
analyze_tf(G)


# ----- G(jw) ------
G_jw = calcola_G_jw(num, den)


# ----- RANGE DI FASE -----
# Calcola la risposta in frequenza
response = ctrl.frequency_response(G, omega)

# Estrai modulo e fase
H = response.frdata[0, 0, :]                # array complesso
phase = np.degrees(np.unwrap(np.angle(H)))  # fase

# In base alla fase in che quadranti siamo ?
print (f"La fase va da {round(min(phase))} a {round(max(phase))}")
quadranti(round(min(phase)), round(max(phase)))


# ----- COMPORTAMENTO ASINTOTICO -----
# Calcola comportamento per w piccolo e per w grande
analisi_limiti(G_jw)

exit(1)

# Aspettare che utente prema invio prima di mostrare i risultati
scelta = input("\nPremi INVIO per vedere i grafici o 'q' per uscire: ")
if scelta.strip().lower() == 'q':
    exit(0)
print()

# Diagramma di Nyquist
ctrl.nyquist_plot(G)
plt.grid(True)
plt.axis('equal')
plt.show()