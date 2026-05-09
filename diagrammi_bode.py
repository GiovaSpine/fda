import numpy as np
import control as ctrl
import matplotlib.pyplot as plt

import numpy as np
import control as ctrl


def my_dcgain(G):
    """
    Trova il guadagno statico
    """

    K = ctrl.dcgain(G)

    if K == np.inf:
        # K come guadagno di Bode: prodotto zeri reali / prodotto poli reali (esclusa origine)
        num_arr = np.array(G.num[0][0], dtype=float)
        den_arr = np.array(G.den[0][0], dtype=float)

        # Rimuovi i fattori s dall'origine dividendo per s^n
        num_trimmed = np.trim_zeros(num_arr[::-1], 'f')[::-1]  # rimuove zeri in coda
        den_trimmed = np.trim_zeros(den_arr[::-1], 'f')[::-1]

        # K è il rapporto dei termini costanti
        K = num_trimmed[-1] / den_trimmed[-1]
        
    return K


def analyze_tf(G):
    z = ctrl.zeros(G)
    p = ctrl.poles(G)
    K = my_dcgain(G)

    if K == np.inf:
        # K come guadagno di Bode: prodotto zeri reali / prodotto poli reali (esclusa origine)
        num_arr = np.array(G.num[0][0], dtype=float)
        den_arr = np.array(G.den[0][0], dtype=float)

        # Rimuovi i fattori s dall'origine dividendo per s^n
        num_trimmed = np.trim_zeros(num_arr[::-1], 'f')[::-1]  # rimuove zeri in coda
        den_trimmed = np.trim_zeros(den_arr[::-1], 'f')[::-1]

        # K è il rapporto dei termini costanti
        K = num_trimmed[-1] / den_trimmed[-1]

    tol = 1e-8

    origin_poles = np.sum(np.isclose(p, 0))

    real_zeros = [x for x in z if abs(x.imag) < tol]
    complex_zeros = [x for x in z if abs(x.imag) >= tol]

    real_poles = [x for x in p if abs(x.imag) < tol]
    complex_poles = [x for x in p if abs(x.imag) >= tol]

    print("Guadagno statico:", K)
    print("Poli nell'origine:", origin_poles)
    print("Zeri reali:", real_zeros)
    print("Zeri complessi:", complex_zeros)
    print("Poli reali:", real_poles)
    print("Poli complessi:", complex_poles)
    print()


def approximated_module(G, omega, stampa=False):
    """
    Calcola il diagramma di Bode approssimato data G funzione di trasferimento e omega,
    che è la lista delle ascisse.
    Ritorna:
        module: lista di valori di ordinata, corrispondente ad ogni valore w di omega
        freq_taglio: lista delle frequenze di taglio
        pendenze: lista che indica la pendenza delle rette, per poterle mostrare nel grafico

    Il parametro stampa se settato su True stampa le frequenze di taglio e smorzamenti sul terminale.
    """

    z = ctrl.zeros(G)
    p = ctrl.poles(G)
    K = my_dcgain(G)

    if K == np.inf:
        # K come guadagno di Bode: prodotto zeri reali / prodotto poli reali (esclusa origine)
        num_arr = np.array(G.num[0][0], dtype=float)
        den_arr = np.array(G.den[0][0], dtype=float)

        # Rimuovi i fattori s dall'origine dividendo per s^n
        num_trimmed = np.trim_zeros(num_arr[::-1], 'f')[::-1]  # rimuove zeri in coda
        den_trimmed = np.trim_zeros(den_arr[::-1], 'f')[::-1]

        # K è il rapporto dei termini costanti
        K = num_trimmed[-1] / den_trimmed[-1]

    tol = 1e-8

    # Separazione origine / reali / complessi
    origin_poles = np.sum(np.isclose(p, 0, atol=tol))

    real_poles = [x for x in p if abs(x.imag) < tol and not np.isclose(x, 0, atol=tol)]
    real_zeros = [x for x in z if abs(x.imag) < tol and not np.isclose(x, 0, atol=tol)]

    complex_poles = [x for x in p if abs(x.imag) >= tol]
    complex_zeros = [x for x in z if abs(x.imag) >= tol]

    # Modulo iniziale(guadagno statico)
    Kabs = abs(K) if abs(K) > tol else tol
    module = np.ones_like(omega) * (20 * np.log10(Kabs))

    # Intervalli di frequenze
    intervalli = [-np.inf]

    # -------------------------
    # Poli all'origine
    # -------------------------
    for _ in range(origin_poles):
        if stampa:
            print(f"- polo nell'origine: 0.0")
            print()

        module += -20 * np.log10(omega)

    # -------------------------
    # Poli reali
    # -------------------------
    for pole in real_poles:
        w0 = abs(np.real(pole))
        intervalli.append(w0)

        if stampa:
            print(f"- polo reale: {pole:.2f}")
            print(f"     frequenza di taglio: {w0:.2f}")
            print()

        for i, w in enumerate(omega):
            if w >= w0:
                module[i] += -20 * np.log10(w / w0)

    # -------------------------
    # Zeri reali
    # -------------------------
    for zero in real_zeros:
        w0 = abs(np.real(zero))
        intervalli.append(w0)

        if stampa:
            print(f"- zero: {zero:.2f}")
            print(f"     frequenza di taglio: {w0:.2f}")
            print()

        for i, w in enumerate(omega):
            if w >= w0:
                module[i] += 20 * np.log10(w / w0)

    # -------------------------
    # Coppie complesse di poli
    # -------------------------
    used = set()
    for i, pole in enumerate(complex_poles):
        if i in used:
            continue

        # Salta il coniugato duplicato
        for j in range(i + 1, len(complex_poles)):
            if np.isclose(pole, np.conj(complex_poles[j])):
                used.add(j)
                break

        wn = abs(pole)
        intervalli.append(wn)
        zeta = -pole.real / wn  # Smorzamento

        if stampa:
            print(f"- polo complesso: {pole:.2f}")
            print(f"     frequenza di taglio: {wn:.2f}")
            print(f"     smorzamento: {zeta:.2f}")
            print()

        for k, w in enumerate(omega):
            if w >= wn:
                module[k] += -40 * np.log10(w / wn)

    # -------------------------
    # Coppie complesse di zeri
    # -------------------------
    used = set()
    for i, zero in enumerate(complex_zeros):
        if i in used:
            continue

        for j in range(i + 1, len(complex_zeros)):
            if np.isclose(zero, np.conj(complex_zeros[j])):
                used.add(j)
                break

        wn = abs(zero)
        intervalli.append(wn)
        zeta = -zero.real / wn  # Smorzamento

        if stampa:
            print(f"- polo complesso: {zero:.2f}")
            print(f"     frequenza di taglio: {wn:.2f}")
            print(f"     smorzamento: {zeta:.2f}")
            print()

        for k, w in enumerate(omega):
            if w >= wn:
                module[k] += 40 * np.log10(w / wn)
    
    # Ricaviamo le pendenze
    intervalli.append(np.inf)
    intervalli = list(np.sort(intervalli))
    pendenze = [0] * (len(intervalli) - 1)

    # Poli all'origine
    for _ in range(origin_poles):
        pendenze = [-20] * (len(intervalli) - 1)
    # Poli reali
    for pole in real_poles:
        w0 = abs(np.real(pole))
        for i in range(len(intervalli) - 1):
            if intervalli[i] >= w0:
                pendenze[i] += -20
    # Zeri reali
    for zero in real_zeros:
        w0 = abs(np.real(zero))
        for i in range(len(intervalli) - 1):
            if intervalli[i] >= w0:
                pendenze[i] += 20
    # Coppie complesse di poli
    used = set()
    for i, pole in enumerate(complex_poles):
        if i in used:
            continue

        # Salta il coniugato duplicato
        for j in range(i + 1, len(complex_poles)):
            if np.isclose(pole, np.conj(complex_poles[j])):
                used.add(j)
                break

        wn = abs(pole)
        for i in range(len(intervalli) - 1):
            if intervalli[i] >= wn:
                pendenze[i] += -40
    # Coppie complesse di zeri
    used = set()
    for i, zero in enumerate(complex_zeros):
        if i in used:
            continue

        for j in range(i + 1, len(complex_zeros) - 1):
            if np.isclose(zero, np.conj(complex_zeros[j])):
                used.add(j)
                break

        wn = abs(zero)
        for i in range(len(intervalli)):
            if intervalli[i] >= wn:
                pendenze[i] += 40

    # Togliendo -inf e inf da intervalli otteniamo la lista
    # delle frequenze di taglio
    freq_taglio = intervalli[1:-1]

    return module, freq_taglio, pendenze


def approximated_phase(G, omega):
    """
    """

    z = ctrl.zeros(G)
    p = ctrl.poles(G)

    tol = 1e-8

    origin_poles = np.sum(np.isclose(p, 0, atol=tol))
    origin_zeros = np.sum(np.isclose(z, 0, atol=tol))

    real_poles = [x for x in p if abs(x.imag) < tol and not np.isclose(x, 0, atol=tol)]
    real_zeros = [x for x in z if abs(x.imag) < tol and not np.isclose(x, 0, atol=tol)]

    complex_poles = [x for x in p if abs(x.imag) >= tol]
    complex_zeros = [x for x in z if abs(x.imag) >= tol]

    phase = np.zeros_like(omega)

    # contributi all'origine
    phase += origin_zeros * 90
    phase -= origin_poles * 90

    logw = np.log10(omega)

    # -----------------
    # zeri reali
    # -----------------
    for zero in real_zeros:
        w0 = abs(np.real(zero))
        a = np.log10(w0) - 1
        b = np.log10(w0) + 1

        for i, lw in enumerate(logw):
            if lw <= a:
                contrib = 0
            elif lw >= b:
                contrib = 90
            else:
                contrib = 45 * (lw - a)

            phase[i] += contrib

    # -----------------
    # poli reali
    # -----------------
    for pole in real_poles:
        w0 = abs(np.real(pole))
        a = np.log10(w0) - 1
        b = np.log10(w0) + 1

        for i, lw in enumerate(logw):
            if lw <= a:
                contrib = 0
            elif lw >= b:
                contrib = -90
            else:
                contrib = -45 * (lw - a)

            phase[i] += contrib

    # -----------------
    # zeri complessi
    # -----------------
    used = set()
    for i, zero in enumerate(complex_zeros):
        if i in used:
            continue

        for j in range(i+1, len(complex_zeros)):
            if np.isclose(zero, np.conj(complex_zeros[j])):
                used.add(j)
                break

        wn = abs(zero)
        a = np.log10(wn) - 1
        b = np.log10(wn) + 1

        for k, lw in enumerate(logw):
            if lw <= a:
                contrib = 0
            elif lw >= b:
                contrib = 180
            else:
                contrib = 90 * (lw - a)

            phase[k] += contrib

    # -----------------
    # poli complessi
    # -----------------
    used = set()
    for i, pole in enumerate(complex_poles):
        if i in used:
            continue

        for j in range(i+1, len(complex_poles)):
            if np.isclose(pole, np.conj(complex_poles[j])):
                used.add(j)
                break

        wn = abs(pole)
        a = np.log10(wn) - 1
        b = np.log10(wn) + 1

        for k, lw in enumerate(logw):
            if lw <= a:
                contrib = 0
            elif lw >= b:
                contrib = -180
            else:
                contrib = -90 * (lw - a)

            phase[k] += contrib

    return phase

# -------------------------------------------------------------------

def show_asymptotic(G, omega):
    module, freq_taglio, pendenze = approximated_module(G, omega, stampa=True)
    phase = approximated_phase(G, omega)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.semilogx(omega, module, linewidth=2)
    ax1.set_title("Bode asintotico")
    ax1.set_ylabel("Modulo [dB]")
    ax1.grid(True, which="both")

    ax2.semilogx(omega, phase, linewidth=2)
    ax2.set_xlabel("ω")
    ax2.set_ylabel("Fase [deg]")
    ax2.grid(True, which="both")

    for w in freq_taglio:
        h = 10  # Altezza dal fondo
        d = 0.6  # Distanza da frequenza
        ax1.axvline(x=w, color='r', linestyle='--', linewidth=1)
        ax1.text(w + w * d, ax1.get_ylim()[0] + h, f'ω={w:.2f}', color='r', fontsize=8, ha='center', va='top')
        ax2.axvline(x=w, color='r', linestyle='--', linewidth=1)
        ax2.text(w + w * d, ax2.get_ylim()[0] + h, f'ω={w:.2f}', color='r', fontsize=8, ha='center', va='top')
    
    # Aggiungi -inf e +inf per avere gli estremi
    edges = [-np.inf] + list(freq_taglio) + [np.inf]

    for i, slope in enumerate(pendenze):
        # Centro dell'intervallo in scala logaritmica
        left = edges[i]
        right = edges[i+1]
        d = 4  # Distanza dalla frequenza
        if np.isinf(left) and np.isinf(right):
            x_mid = omega[len(omega)//2]
        elif np.isinf(left):
            x_mid = right / d
        elif np.isinf(right):
            x_mid = left * d
        else:
            x_mid = np.sqrt(left * right)  # Centro geometrico

        # Ordinata: valore del modulo in x_mid
        y_mid = np.interp(x_mid, omega, module)

        h = -8  # Altezza dalla retta
        label = f"{slope:+.0f} dB/dec" if slope != 0 else "0 dB/dec"
        ax1.text(x_mid, y_mid + h, label, color='blue', fontsize=8, ha='center')

    plt.tight_layout()
    plt.show()

def show_real(G, omega):

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # -------------------------
    # dati reali dal sistema
    # -------------------------
    mag, phase, _ = ctrl.frequency_response(G, omega)

    real_module = 20 * np.log10(np.abs(mag))
    real_phase = np.degrees(np.unwrap(phase))

    # -------------------------
    # approssimato (per allineamento)
    # -------------------------
    approx_phase = approximated_phase(G, omega)

    # -------------------------
    # ALIGN FASE (multipli di 360°)
    # -------------------------
    offset = np.mean(real_phase - approx_phase)
    offset = 360 * np.round(offset / 360)

    real_phase = real_phase - offset

    # -------------------------
    # MODULO
    # -------------------------
    ax[0].semilogx(omega, real_module)
    ax[0].set_ylabel("Modulo [dB]")
    ax[0].grid(True, which="both")
    ax[0].set_title("Bode reale (allineato)")

    # -------------------------
    # FASE
    # -------------------------
    ax[1].semilogx(omega, real_phase)
    ax[1].set_xlabel("ω [rad/s]")
    ax[1].set_ylabel("Fase [deg]")
    ax[1].grid(True, which="both")

    plt.tight_layout()
    plt.show()

def show_both(G, omega):

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # reale
    mag, phase, _ = ctrl.frequency_response(G, omega)
    real_module = 20 * np.log10(np.abs(mag))
    real_phase = np.degrees(np.unwrap(phase))

    # approssimato
    approx_module, _, _ = approximated_module(G, omega)
    approx_phase = approximated_phase(G, omega)

    # ALIGN FASE
    offset = np.mean(real_phase - approx_phase)
    offset = 360 * np.round(offset / 360)
    real_phase = real_phase - offset

    # -----------------------
    # MODULO
    # -----------------------
    ax[0].semilogx(
        omega,
        real_module,
        "--",
        color="tab:blue",
        label="Reale"
    )

    ax[0].semilogx(
        omega,
        approx_module,
        color="tab:red",
        label="Asintotico"
    )

    ax[0].set_ylabel("Modulo [dB]")
    ax[0].grid(True, which="both")
    ax[0].legend()

    # -----------------------
    # FASE
    # -----------------------
    ax[1].semilogx(
        omega,
        real_phase,
        "--",
        color="tab:blue",
        label="Reale (allineato)"
    )

    ax[1].semilogx(
        omega,
        approx_phase,
        color="tab:red",
        label="Asintotico"
    )

    ax[1].set_xlabel("ω [rad/s]")
    ax[1].set_ylabel("Fase [deg]")
    ax[1].grid(True, which="both")
    ax[1].legend()

    plt.show()

# ===================================================================

# P(s) = N(s) / D(s)
# Dove N(s) e D(s) sono polinomi qualsiasi (non per forza monico)

# es. 2s^2 + 3 diventa [2, 0, 3]

# Puoi fare anche così
# denominatore: (s+1)(s^2 + 20s + 400)
# den1 = [1, 1]
# den2 = [1, 20, 400]
# den = np.convolve(den1, den2)

num = [1, 20, 100]
den = [1, 200, 10000, 0]

num = [200, 20]
den = [1/400, 1/400 + 1/20, 1/20 + 1, 1]

num = [3.6]
den = [1, 4, 9, 0]

G = ctrl.TransferFunction(num, den)

# Stampa qualche informazione sulla G(s)
print(G, "\n")
analyze_tf(G)

# Dominio personalizzato(rad/s)
# es. np.logspace(-2, 3, 1000) va da 10^-2 a 10^3
omega = np.logspace(-2, 4, 1000)

# Asintotico
show_asymptotic(G, omega)

# Reale
show_real(G, omega)

# Confronto
show_both(G, omega)
