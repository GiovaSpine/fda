import control as ctrl
import numpy as np


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

    print(f"Guadagno statico: {K:.3f}")
    print(f"\nPoli nell'origine: {origin_poles}")
    print(f"\nZeri reali:")
    for zero in real_zeros:
        print(f"   {zero:.3f}")
    print(f"\nZeri complessi")
    for zero in complex_zeros:
        print(f"   {zero:.3f}")
    print(f"\nPoli reali")
    for pole in real_poles:
        print(f"   {pole:.3f}")
    print(f"\nPoli complessi")
    for pole in complex_poles:
        print(f"   {pole:.3f}")
    print()