import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
import sympy as sp
from collections import Counter
from itertools import combinations


def roots_cleaning(roots, tol=1e-3, digits=6):
    roots = np.asarray(roots, dtype=complex)
    groups = []

    # raggruppamento
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

        # mantiene molteplicità
        cleaned.extend([val] * len(g))

    return np.array(cleaned, dtype=complex)


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
        sumz += np.angle(z - p, deg=True)
    for zz in zeros:
        if abs(zz - z) > 1e-8:
            sump += np.angle(z - zz, deg=True)
    theta = -180 - sumz + sump
    while theta > 180:
        theta -= 360
    while theta < -180:
        theta += 360

    return theta


def angolo_di_partenza_inverso(p, poles, zeros):
    return angolo_di_partenza_diretto(p, poles, zeros) - 180.0

def angolo_di_arrivo_inverso(z, poles, zeros):
    return angolo_di_arrivo_diretto(z, poles, zeros) - 180.0


def annotated_root_locus_plot(
        title,
        G,
        poles,
        zeros,
        centroid,
        asym_angles,
        breaks,
        angolo_di_partenza,
        angolo_di_arrivo
        ):

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

    # CENTRO DEGLI ASINTOTI
    if centroid is not None:
        ax.plot(
            centroid,
            0,
            'ro',
            markersize=7,
            label='Centro degli asintoti'
        )

        # asintoti
        L = 200

        for theta in asym_angles:
            th = np.deg2rad(theta)

            x = [centroid, centroid + L*np.cos(th)]
            y = [0, L*np.sin(th)]

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
            -0.5,
            0.5,
            colors="#FF9900",
            linewidth=3,
            label='Punto multiplo'
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
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_title(title)

    plt.show()



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
    asym_angles = []
    if q > 0:
        centroid = (np.sum(poles).real - np.sum(zeros).real) / q

        print(f"\nNumero asintoti = {q}")
        print(f"Centro degli asintoti = {centroid:.4f}")

        print(f"Angoli tra asintoti: {360.0 / q}")
        print("Angoli asintoti:")
        for k in range(q):
            theta = (2*k + 1) * 180 / q
            asym_angles.append(theta)
            print(f"   {theta:.2f}°")

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

        print(f"   {b:.3f};  molteplicità: {breaks.get(b)};  appartiene al luogo {direzione_luogo}")

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

    annotated_root_locus_plot(
        "Luogo delle radici diretto (k>0)",
        G,
        poles,
        zeros,
        centroid,
        asym_angles,
        direct_breaks,
        angolo_di_partenza_diretto,
        angolo_di_arrivo_diretto
    )
    annotated_root_locus_plot(
        "Luogo delle radici inverso (k<0)",
        -G,
        poles,
        zeros,
        centroid,
        asym_angles, 
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

G = ctrl.TransferFunction(num, den)

print()
punti_multipli(G)



#fig, ax = plt.subplots(figsize=(8,8))
ctrl.root_locus(G, grid=True)
plt.show()

#fig, ax = plt.subplots(figsize=(8,8))
#ctrl.root_locus_plot(
#    -G,
#    ax=ax,
#    color='red'
#)
#plt.show()

annotated_root_locus(G)  # seconda finestra annotata

