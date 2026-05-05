import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
import sympy as sp
from collections import Counter
from itertools import combinations


def roots_cleaning(roots, tol=1e-3, digits=6):
    roots = np.asarray(roots, dtype=complex)
    groups = []

    # Raggruppamento
    for r in roots:
        inserted = False

        for g in groups:
            center = sum(g) / len(g)

            if abs(r - center) < tol:
                g.append(r)
                inserted = True
                break

        if not inserted:
            groups.append([r])

    cleaned = []

    # media + pulizia
    for g in groups:
        center = sum(g) / len(g)

        re = round(center.real, digits)
        im = round(center.imag, digits)

        if abs(im) < tol:
            im = 0.0

        val = complex(re, im)

        # Mantiene molteplicità
        cleaned.extend([val] * len(g))

    return np.array(cleaned, dtype=complex)

# ---------------------------------------------------------

def punti_multipli(G):
    s = sp.Symbol('s')

    num, den = ctrl.tfdata(G)
    num = np.atleast_1d(np.squeeze(num))
    den = np.atleast_1d(np.squeeze(den))

    N = sum(c * s**(len(num)-i-1) for i, c in enumerate(num))
    D = sum(c * s**(len(den)-i-1) for i, c in enumerate(den))

    K = -D / N

    dK = sp.diff(K, s)

    eq = sp.simplify(sp.together(dK))

    num_eq, _ = sp.fraction(eq)

    poly = sp.Poly(sp.expand(num_eq), s)

    roots = poly.nroots()

    # ATTENZIONE
    # Tra i punti multipli possono esserci poli e zeri dell'anello aperto,
    # se questi hanno molteplicità > 1
    # Li togliamo, perché li gestiamo a parte
    poles = np.array(ctrl.poles(G))
    zeros = np.array(ctrl.zeros(G))

    singular = np.concatenate([poles, zeros])
    TOL=0.0001
    roots = [r for r in roots if not any(abs(complex(r) - complex(p)) < TOL for p in singular)]
        
    # Ignoriamo le radici complesse
    # e contiamo le molteplicità
    breaks = {}

    for r in roots:
        if abs(sp.im(r)) > TOL:
            continue
        r_real = float(sp.re(r))
        
        # Cerca se esiste già una chiave vicina
        found = False
        for key in breaks:
            if abs(r_real - key) < TOL:
                breaks[key] += 1
                found = True
                break
        
        if not found:
            breaks[r_real] = 1
    
    return breaks

# ---------------------------------------------------------

def angolo_di_partenza_diretto(p, poles, zeros):
    """
    formula classica angolo di partenza
    """
    sumz = 0
    sump = 0
    for z in zeros:
        sumz += np.angle(p - z, deg=True)
    for pp in poles:
        if abs(pp - p) > 1e-8:
            sump += np.angle(p - pp, deg=True)
    theta = 180 + sumz - sump
    while theta > 180:
        theta -= 360
    while theta < -180:
        theta += 360
    return theta

def angolo_di_arrivo_diretto(z, poles, zeros):
    """
    formula classica angolo di arrivo
    """
    sumz = 0
    sump = 0
    for p in poles:
        sump += np.angle(z - p, deg=True)
    for zz in zeros:
        if abs(zz - z) > 1e-8:
            sumz += np.angle(z - zz, deg=True)
    theta = 180 + sump - sumz
    while theta > 180:
        theta -= 360
    while theta < -180:
        theta += 360
    return theta

def angolo_di_partenza_inverso(p, poles, zeros):
    return angolo_di_partenza_diretto(p, poles, zeros) - 180.0

def angolo_di_arrivo_inverso(z, poles, zeros):
    return angolo_di_arrivo_diretto(z, poles, zeros) - 180.0

# ---------------------------------------------------------

def radici_multiple_plot(ax, direct, poles, zeros):
    # POLI E ZERI (ANELLO APERTO) MULTIPLI
    # Dividono il piano in un numero di parti equiangole e simmetriche
    # pari alla molteplicità

    # ATTENZIONE
    # Per semplicita consideriamo poli e zeri reali
    # (non so se può dare poli complessi coniugati a molteplicità maggiore di 1)

    # Come facciamo a riconoscere come divide il piano ?
    # dobbiamo sfruttare la conoscenza dell'asse reale
    poles_dict = dict(Counter(poles))
    zeros_dict = dict(Counter(zeros))
    poles_dict = {k: v for k, v in poles_dict.items() if v > 1 and v.imag == 0}
    zeros_dict = {k: v for k, v in zeros_dict.items() if v > 1 and v.imag == 0}

    asse_reale = [np.real(s) for s in list(poles) + list(zeros)]
    asse_reale.append(-np.inf)
    asse_reale.append(np.inf)
    asse_reale = np.sort(asse_reale)

    for p in poles_dict.keys():
        for i in range(len(asse_reale)):
            if p.real < asse_reale[i]:
                right_roots = len(asse_reale) - 1 - i
                break
        
        pole_thetas = None

        if direct:
            if right_roots % 2 != 0:
                # Allora il polo si sposta a destra
                pole_thetas = [0.0]
                for i in range(1, poles_dict.get(p)):
                    pole_thetas.append(pole_thetas[i-1] + (360.0 / poles_dict.get(p)))
            if (right_roots + poles_dict.get(p)) % 2 != 0:
                # Allora il polo si sposta a sinistra
                pole_thetas = [180.0]
                for i in range(1, poles_dict.get(p)):
                    pole_thetas.append(pole_thetas[i-1] + (360.0 / poles_dict.get(p)))
        else:
            if right_roots % 2 == 0:
                # Allora il polo si sposta a destra
                pole_thetas = [0.0]
                for i in range(1, poles_dict.get(p)):
                    pole_thetas.append(pole_thetas[i-1] + (360.0 / poles_dict.get(p)))
            if (right_roots + poles_dict.get(p)) % 2 == 0:
                # Allora il polo si sposta a sinistra
                pole_thetas = [180.0]
                for i in range(1, poles_dict.get(p)):
                    pole_thetas.append(pole_thetas[i-1] + (360.0 / poles_dict.get(p)))
        
        if pole_thetas is None:
            if poles_dict.get(p) % 2 == 0:
                # Siamo in grado di dividere il piano in parti uguali
                # perché sappiamo che deve essere simmetrico e non va né a destra né a sinistra
                pole_thetas = [180.0 / poles_dict.get(p)]
                for i in range(1, poles_dict.get(p)):
                    pole_thetas.append(pole_thetas[i-1] + (360.0 / poles_dict.get(p)))
            else:
                print(f"ATTENZIONE: non siamo in grado di dividere il piano in {poles_dict.get(p)} part")
                continue
            
        for theta in pole_thetas:
            th = np.deg2rad(theta)

            length = 0.8

            # vettore direzione normalizzato
            dx = length*np.cos(th)
            dy = length*np.sin(th)

            ax.annotate(
                '',
                xy=(p + 1.2*dx, 0.0 + 1.2*dy),
                xytext=(p, 0.0),
                arrowprops=dict(
                    arrowstyle='->',
                    color='green',
                    lw=1.5,
                    shrinkA=0,
                    shrinkB=0,
                    label="Direzione partenza del polo"
                ),
                zorder=5
            )

def annotated_root_locus_plot(
        direct,
        G,
        poles,
        zeros,
        centroid,
        asym_angle,
        breaks,
        angolo_di_partenza,
        angolo_di_arrivo
        ):
    """
    direct è True o False: indica se è diretto o inverso
    """
    
    n = len(poles)
    m = len(zeros)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, which='both', linestyle='--', linewidth=0.6)
    ax.axhline(0, color="#A1A1A1FF", linewidth=1)  # asse reale
    ax.axvline(0, color='#A1A1A1FF', linewidth=1)  # asse immaginario

    ctrl.root_locus_plot(G, grid=True, ax=ax, label='Luogo delle radici')

    # POLI
    for p in poles:
        ax.plot(
            p.real,
            p.imag,
            'bx',
            markersize=7,
            zorder=20
        )

    # ZERI
    if zeros is not None and len(zeros) > 0:
        for z in zeros:
            ax.plot(
                z.real,
                z.imag,
                marker='o',
                markersize=7,
                markerfacecolor='none',  # vuoto -> anello
                markeredgecolor='blue',
                linewidth=2,
                zorder=20
            )
    
       
    radici_multiple_plot(ax, direct, poles, zeros)

    # CENTRO DEGLI ASINTOTI
    if centroid is not None:
        ax.plot(
            centroid,
            0,
            'ro',
            markersize=7,
            label='Centro degli asintoti'
        )

        # ASINTOTI
        # Come si ricavano? dipende da asse reale:
        # se so che un semiasse reale è asintoto e l'angolo tra essi (asym_angle)
        # allora li so disegnare

        q = n - m

        length = 200

        # Controlliamo se un semiasse reale è asintoto
        if direct:
            # Nel caso di luogo delle radici diretto
            asym_angles = []
            for k in range(q):
                theta = (2*k + 1) * 180 / q
                asym_angles.append(theta)
        else:
            # Nel caso di luogo delle radici inverso
            # il semiasse reale verso +infinito è sempre asintoto
            asym_angles = [0.0]
            for i in range(1, q):
                asym_angles.append(asym_angles[i-1] + asym_angle)

        for theta in asym_angles:
            if theta % 180.0 == 0:
                # Non disegniamo gli asintitoti piatti dell'asse reale
                continue

            th = np.deg2rad(theta)

            x = [centroid, centroid + length*np.cos(th)]
            y = [0, length*np.sin(th)]

            ax.plot(
                x,
                y,
                'r--',
                linewidth=1.5
            )

    # PUNTI MULTIPLI
    for b in breaks.keys():
        ax.vlines(
            b,
            -0.3,
            0.3,
            colors="#E46E00",
            linewidth=3,
            label='Punto multiplo',
            zorder=10
        )

        # i punti multipli dividono il piano in numero poli * molteplicita
        plane_divisions = 2 * (breaks.get(b) + 1)
        for theta in np.linspace(360.0 / plane_divisions, 360.0, plane_divisions, True):
            th = np.deg2rad(theta)

            length = 0.8

            # vettore direzione normalizzato
            dx = length*np.cos(th)
            dy = length*np.sin(th)

            ax.annotate(
                '',
                xy=(b + 1.2*dx, 0.0 + 1.2*dy),
                xytext=(b, 0.0),
                arrowprops=dict(
                    arrowstyle='->',
                    color='orange',
                    lw=1.5,
                    shrinkA=0,
                    shrinkB=0,
                    label="Divisione del punto multiplo"
                ),
                zorder=5
            )


    # FRECCE DI PARTENZA
    for p in poles:
        if p.imag != 0.0:

            theta = angolo_di_partenza(p, poles, zeros)
            th = np.deg2rad(theta)

            length = 1.5

            # vettore direzione normalizzato
            dx = length*np.cos(th)
            dy = length*np.sin(th)

            ax.annotate(
                '',
                xy=(p.real + 1.2*dx, p.imag + 1.2*dy),
                xytext=(p.real, p.imag),
                arrowprops=dict(
                    arrowstyle='->',
                    color='red',
                    lw=1.8,
                    shrinkA=0,
                    shrinkB=0
                ),
                zorder=10
            )
            
            # Vettore riferimento per capire angolo
            ax.plot(
                [p.real, p.real + length],
                [p.imag, p.imag],
                'r--',
                linewidth=1.5
            )


    # FRECCE DI ARRIVO
    for z in zeros:
        if z.imag != 0.0:

            theta = angolo_di_arrivo(z, poles, zeros)
            th = np.deg2rad(theta)

            length = 1.5

            # vettore direzione normalizzato
            dx = length*np.cos(th)
            dy = length*np.sin(th)

            ax.annotate(
                '',
                xy=(z.real, z.imag),
                xytext=(z.real + 1.2*dx, z.imag + 1.2*dy),
                arrowprops=dict(
                    arrowstyle='->',
                    color='red',
                    lw=1.8,
                    shrinkA=0,
                    shrinkB=0
                ),
                zorder=10
            )
            
            # Vettore riferimento per capire angolo
            ax.plot(
                [z.real, z.real + length],
                [z.imag, z.imag],
                'r--',
                linewidth=1.5
            )


    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    if direct:
        ax.set_title("Luogo delle radici diretto (k>0)")
    else:
        ax.set_title("Luogo delle radici inverso (k<0)")


    plt.show()


# ---------------------------------------------------------

def annotated_root_locus(G):

    # ----- SINGOLARITA' -----
    poles = np.array(ctrl.poles(G))
    zeros = np.array(ctrl.zeros(G))

    # poles e zeros possono avere imprecisioni
    # del tipo -5.000026 e -4.999947 che sono la stessa radice
    poles = roots_cleaning(poles)
    zeros = roots_cleaning(zeros)

    n = len(poles)
    m = len(zeros)

    print("\n==============================")
    print("      LUOGO DELLE RADICI")
    print("==============================")

    print("\nPoli:")
    for p in poles:
        print(" ", p)

    print("\nZeri:")
    for z in zeros:
        print(" ", z)

    # ----- ASSE REALE -----
    # per determinare quale x reale appartiene a luogo diretto e inverso
    asse_reale = [np.real(s) for s in list(poles) + list(zeros)]
    asse_reale.append(-np.inf)
    asse_reale.append(np.inf)
    asse_reale = np.sort(asse_reale)

    # ----- ASINTOTI -----
    q = n - m

    centroid = None
    asym_angle = 360.0 / q
    if q > 0:
        centroid = (np.sum(poles).real - np.sum(zeros).real) / q

        print(f"\nNumero asintoti = {q}")
        print(f"Centro degli asintoti = {centroid:.4f}")

        print(f"Angoli tra asintoti: {asym_angle}")

    # ----- PUNTI MULTIPLI -----
    breaks = punti_multipli(G)

    direct_breaks = {}
    inverse_breaks = {}

    print("\nPunti multipli:")
    for b in breaks.keys():
        # Verifichiamo se appartiene al luogo diretto (k>0) o inverso (k<0)
        for i in range(len(asse_reale) - 1):
            if b > asse_reale[i] and b <= asse_reale[i+1]:
                if (len(asse_reale) - i - 2) % 2 != 0:  # -2 perché -inf e inf non sono singolarità
                    # Allora b appartiene al luogo diretto
                    direct_breaks[b] = breaks[b]
                    direzione_luogo = "diretto"
                else:
                    # Allora b appartiene al luogo inverso
                    inverse_breaks[b] = breaks[b]
                    direzione_luogo = "inverso"

        print(f"   {b:.3f};  molteplicità: {breaks.get(b)};  appartiene al luogo {direzione_luogo};")
        print(f"   Divide il piano in 2 * (molteplicità + 1) = {2 * (breaks.get(b) + 1)} parti (angolo: {360.0 / (2 * (breaks.get(b) + 1))})")
        print()

    # ----- POLI E ZERI (ANELLO APERTO) MULTIPLI -----


    # ----- ANGOLI DI PARTENZA -----
    # Se si ha dei poli complessi
    if any(np.imag(poles) != 0.0):
        print("\nAngoli di partenza nel luogo diretto (k>0):")
        for p in poles[np.imag(poles) != 0.0]:
            theta = angolo_di_partenza_diretto(p, poles, zeros)
            print(f"   polo {p} -> {theta:.2f}°")
        print("\nAngoli di partenza nel luogo inverso (k<0):")
        for p in poles[np.imag(poles) != 0.0]:
            theta = angolo_di_partenza_inverso(p, poles, zeros)
            print(f"   polo {p} -> {theta:.2f}°")

    # ----- ANGOLI DI ARRIVO -----
    # Se si ha degli zeri complessi
    if any(np.imag(zeros) != 0.0):
        print("\nAngoli di arrivo nel luogo diretto (k>0):")
        for z in zeros[np.imag(zeros) != 0.0]:
            theta = angolo_di_arrivo_diretto(z, poles, zeros)
            print(f"   zero {z} -> {theta:.2f}°")
        print("\nAngoli di arrivo nel luogo inverso (k<0):")
        for z in zeros[np.imag(zeros) != 0.0]:
            theta = angolo_di_arrivo_inverso(z, poles, zeros)
            print(f"   zero {z} -> {theta:.2f}°")
    print()

    # ----- INTERESEZIONE ASSE IMMAGINARIO -----


    # ----- PLOT -----
    # Luogo diretto
    annotated_root_locus_plot(
        True,
        G,
        poles,
        zeros,
        centroid,
        asym_angle,
        direct_breaks,
        angolo_di_partenza_diretto,
        angolo_di_arrivo_diretto
    )
    # Luogo inverso
    annotated_root_locus_plot(
        False,
        -G,
        poles,
        zeros,
        centroid,
        asym_angle, 
        inverse_breaks,
        angolo_di_partenza_inverso,
        angolo_di_arrivo_inverso
    )

    

# ===================================================================

# data G(s) espressa come divisione di polinomi
# es. den = [1, 3, 2, 0] è s^3 + 3s^2 + 2s
num = [1]
den = [1, 3, 2, 0]

#num = [1, 1]
#den = [1, 9, 0, 0]

#num = [1, 5]
#den = [1, 6, 109, 0]

num = [1, 20, 101]
den = [1, 15, 75, 125, 0]

num = [1, 2, 1]
den = [1, 200, 10000, 0]

G = ctrl.TransferFunction(num, den)

print()
punti_multipli(G)



#fig, ax = plt.subplots(figsize=(8,8))
ctrl.root_locus(G, grid=True)
plt.show()

ctrl.root_locus(-G, grid=True)
plt.show()

#fig, ax = plt.subplots(figsize=(8,8))
#ctrl.root_locus_plot(
#    -G,
#    ax=ax,
#    color='red'
#)
#plt.show()

annotated_root_locus(G)  # seconda finestra annotata

