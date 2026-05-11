import control as ctrl
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from utily import *


def percorso_nyquist(G):
    p = ctrl.poles(G)

    tol = 1e-8

    origin_poles = np.sum(np.isclose(p, 0))
    imag_poles = np.sort([x.imag for x in p if np.isclose(x.real, 0) and abs(x.imag) >= tol])
    n_imag = len(imag_poles)

    # Il percorso di Nyquist cambia se si hanno poli nell'origine o immaginari puri

    print("Percorso di Nyquist")
    if origin_poles > 0 and n_imag == 0:
        print("w segue il percorso: 0+ => +oo => -oo => 0-")

    elif origin_poles == 0 and n_imag!= 0:
        print("w segue il percorso: 0 => ", end='')
        for i in range(int(n_imag / 2), n_imag):
            print(f"{imag_poles[i]:.2f}- =>  {imag_poles[i]:.2f}+ => ", end='')
        print("+oo => -oo => ", end='')
        for i in range(0, int(n_imag / 2)):
            print(f"{imag_poles[i]:.2f}- =>  {imag_poles[i]:.2f}+ => ", end='')
        print("0")
                
    elif origin_poles > 0 and len(imag_poles) == 0:
        print("w segue il percorso: 0+ => ", end='')
        for i in range(int(n_imag / 2), n_imag):
            print(f"{imag_poles[i]:.2f}- =>  {imag_poles[i]:.2f}+ => ", end='')
        print("+oo => -oo => ", end='')
        for i in range(0, int(n_imag / 2)):
            print(f"{imag_poles[i]:.2f}- =>  {imag_poles[i]:.2f}+ => ", end='')
        print("-0")
    else:
        print("w segue il percorso: 0 => +oo => -oo => 0")
    print()


def get_phase_range(G, omega):
    """
    Ricava il range delle fase che attraversa la risposta in frequenza, stando
    attenta alle discontinuità.

    Es. per G(s) = 1 / ((s + 1)*(s + 1) ^ 2)
    si ha poli immaginari puri, per cui lo smorzamento è 0 e si ha
    discontinuità nella fase.
    """

    # Calcola la risposta in frequenza
    _, phase, _ = ctrl.frequency_response(G, omega)
    real_phase = np.degrees(np.unwrap(phase))
    #real_phase = np.degrees(phase)

    phases = []
    aux = []

    # Soglia tra due valori di fase che se superata indica discontinuità
    THRESHOLD = 5.0

    for i in range(len(real_phase) - 1):
        if np.isnan(real_phase[i]) or abs(real_phase[i] - real_phase[i+1]) > THRESHOLD:
            print("Discontinuità:", real_phase[i], real_phase[i+1])
            phases.append(aux)
            aux = []
        else:
            aux.append(real_phase[i])
    phases.append(aux)
    
    return [(min(p), max(p)) for p in phases]


def quadranti(phase_range):
    """
    Stampa in quali quadranti si trova il diagramma di Nyquist in base alla fase.
    Si ricorda che:
    
        secondo | primo
        --------|--------
         terzo  | quarto
    """
    
    def normalizza_fase(phase):
        # porta in [0, 360)
        a = phase % 360 
        return a
    
    quadranti = set()
    
    for min_phase, max_phase in phase_range:
        
        min_phase = normalizza_fase(min_phase)
        max_phase = normalizza_fase(max_phase)

        if max_phase <= 90:
            # "primo"
            quadranti.add("primo")
        elif max_phase <= 180.0:
            if min_phase >= 90.0:
                # "secondo"
                quadranti.add("secondo")
            else:
                # "primo e secondo"
                quadranti.add("primo")
                quadranti.add("secondo")
        elif max_phase <= 270.0:
            if min_phase >= 180.0:
                # "terzo"
                quadranti.add("terzo")
            elif min_phase >= 90.0:
                # "secondo e terzo"
                quadranti.add("secondo")
                quadranti.add("terzo")
            else:
                # "primo, secondo e terzo"
                quadranti.add("primo")
                quadranti.add("secondo")
                quadranti.add("terzo")
        else:
            if min_phase >= 270.0:
                # "quarto"
                quadranti.add("quarto")
            elif min_phase >= 180.0:
                # "terzo e quarto"
                quadranti.add("terzo")
                quadranti.add("quarto")
            elif min_phase >= 90.0:
                # "secondo, terzo e quarto"
                quadranti.add("secondo")
                quadranti.add("terzo")
                quadranti.add("quarto")
            else:
                # "primo, secondo, terzo e quarto"
                quadranti.add("primo")
                quadranti.add("secondo")
                quadranti.add("terzo")
                quadranti.add("quarto")

    print(f"Per w che va da 0 all'infinito, siamo nei quadranti: {quadranti}")
    print("Si ricorda che:\n\n"
        "secondo | primo\n"
        "--------|--------\n"
        " terzo  | quarto\n"
          )
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


def analisi_limiti(G_jw, num, den):
    """
    Data G(jw), num e den calcola i limiti per w->0 e w->inf, usando sympy
    e i termini dominanti per w->0 e w->inf.
    """
    
    w = sp.symbols('w', real=True)

    n = len(den) - 1  # grado denominatore
    m = len(num) - 1  # grado numeratore

    # Esiste un asintoto ? 
    asintoto_exist = False

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
        asintoto_exist = True
    else:
        # Tende a 0
        # Non credo posso accadere
        raise ValueError("ERRORE: per w -> 0, G(jw) -> 0. Significherebbe un diagramma di Bode che scende all'infinito a sinistra")
    print()

    # Limite w -> inf
    # Si comporta circa come 1 / (jw)^(n-m)
    # (allora l'ampiezza è sempre zero ? dipende se m < n)
    print(f"Limite w --> inf = ", end='')
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
    print()
    
    if asintoto_exist:
        print("Poiché l'ampiezza tende all'infinito per w -> 0 si ha un asintoto")
        # Separa parte reale e immaginaria
        expr_expanded = sp.simplify(G_jw)
        re = sp.re(expr_expanded)
        im = sp.im(expr_expanded)

        re_simplified = sp.simplify(re)
        im_simplified = sp.simplify(im)
        print(f"G(jw) = a(w)+ jb(w) = ")
        print(f"      = ({re_simplified}) + j*({im_simplified})")
        print()
        print(f"   Re = {re_simplified}")
        print(f"   Im = {im_simplified}")
        print()

        lim_re_0 = sp.limit(re_simplified, w, 0)
        lim_re_0 = sp.simplify(lim_re_0)
        lim_im_0 = sp.limit(im_simplified, w, 0)
        lim_im_0 = sp.simplify(lim_im_0)
        print(f"Re{{G(j*0)}} = {lim_re_0}")
        print(f"Im{{G(j*0)}} = {lim_im_0}")
        print()

        if abs(lim_re_0) != sp.oo and abs(lim_im_0) == sp.oo:
            print(f"Asintoto reale in {lim_re_0} = {lim_re_0.evalf(4)}")
        elif abs(lim_im_0) != sp.oo and abs(lim_re_0) == sp.oo:
            print(f"Asintoto immaginario in {lim_im_0} = {lim_im_0.evalf(4)}")
        elif abs(lim_re_0) == sp.oo and abs(lim_im_0) == sp.oo:
            # E' possibile che tendano entrambi all'infinito
            # es. 1 / (s^2(s + 1))
            # Si risolve con la sostituzione:
            #   x = Re(G(jw)) = g1(w)
            #   y = Im(G(jw)) = g2(w)
            #   trovi w in funzione di y da g2(w)
            #   sostituisci w con espressione di y in x
            #   trovi x = g3(y)
            print("Tendono entrambi a infinito")
            print("Semplifichiamo le epressioni per w -> 0, sennò otteniamo espressioni complicate")
            print("(sapendo che w^(k+1) va a 0 più in fretta di w^k per w -> 0)")
            print()

            w, x, y = sp.symbols('w x y', real=True)

            # Termine dominante per w->0 (grado minimo)
            re_dom = sp.series(re_simplified, w, 0, n=1).as_leading_term(w)
            im_dom = sp.series(im_simplified, w, 0, n=1).as_leading_term(w)

            print(f"Re (dominante) = {re_dom}")
            print(f"Im (dominante) = {im_dom}")

            # Sostituzione per ricavare x = f(y)
            w_sol = sp.solve(sp.Eq(y, im_dom), w)[0]
            curva = sp.simplify(re_dom.subs(w, w_sol))
            print(f"L'asintoto è x = f(y) = {curva}")
        else:
            # Impossibile, almeno uno deve tendere all'infinito se G(j0) tende all'infinito
            raise ValueError("ERRORE durante il calcolo dell'asintoto")
        
        print()

    return G_jw, lim0, lim_inf


# ===================================================================

# P(s) = N(s) / D(s)
# Dove N(s) e D(s) sono polinomi qualsiasi (non per forza monico)

# es. 2s^2 + 3 diventa [2, 0, 3]

num = [3.6]
den = [1, 4, 9, 0]

num = [1]
den = [1, 1, 0, 0]

num = [1]
den = [1, 1, 1, 1]

num = [1]
den = [1, 3, 2, 0]


G = ctrl.TransferFunction(num, den)


# Range di frequenze
# es. np.logspace(-2, 3, 1000) va da 10^-2 a 10^3
# può essere necessario ingrandirlo, perché serve per trovare il range della fase
# (si suggerisce di fare i diagrammi di Bode e guardare il range di frequenze, se necessario)
LEN = 5000
omega = np.logspace(-5, 5, LEN)

# ---------------------------------

print(G, "\n")
analyze_tf(G)


# ----- G(jw) ------
G_jw = calcola_G_jw(num, den)


# ----- PERCORSO DI NYQUIST -----
percorso_nyquist(G)


# ----- RANGE DI FASE -----
phase_range = get_phase_range(G, omega)

print("La fase, per w che va da 0 all'infinito, va da:")
for min_p, max_p in phase_range:
    print(f"   {round(min_p)} a {round(max_p)}") 
print()

# In base alla fase in che quadranti siamo ?
# (per w che va da 0 a infinito)
quadranti(phase_range)


# ----- COMPORTAMENTO ASINTOTICO -----
# Calcola comportamento per w piccolo e per w grande
analisi_limiti(G_jw, num, den)


# Aspettare che utente prema invio prima di mostrare i risultati
scelta = input("Premi INVIO per vedere i grafici o 'q' per uscire: ")
if scelta.strip().lower() == 'q':
    exit(0)
print()

# Diagramma di Nyquist
ctrl.nyquist_plot(G)
plt.grid(True)
plt.axis('equal')
plt.show()