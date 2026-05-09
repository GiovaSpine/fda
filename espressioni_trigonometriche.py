import sympy as sp

# Utile per Comunicazioni Numeriche:
# se provi a risolvere uno stesso esercizio TCF (Trasfromata Continua di Fourier)
# in diversi modi, otterrai espressioni diverse, che sono complicate da verificare
# se sono uguali, usando regole trigonometriche.
# Con questo codice è possibile inserire le espressioni e valutare se sono identiche.

# ---------------------------------------------------------

# SIMBOLI E FUNZIONI

f, A, T, k, B = sp.symbols('f A T k B', real=True)

def sinc(f, T):
    # espressione: sinc(fT) = sin(pi * f * T) / (pi * f * T)
    return sp.sin(sp.pi * f * T) / (sp.pi * f * T)

# ---------------------------------------------------------

# COMPARATORE

def equal(esp1, esp2):
    # Riscrivo esp1 in termini di cos
    esp1 = esp1.rewrite(sp.cos)
    esp2 = esp2.rewrite(sp.cos)

    # Semplifico tutto
    esp1_simpl = sp.trigsimp(sp.simplify(esp1))
    esp2_simpl = sp.trigsimp(sp.simplify(esp2))

    # Confronto
    diff = sp.simplify(esp1_simpl - esp2_simpl)

    return diff == 0

# ---------------------------------------------------------

"""
=================================
REGOLE PER SCRIVERE LE EPRESSIONI
=================================

COSTANTI IMPORTANTI
sp.I        unità immaginaria
sp.pi       pi greco
sp.E        numero di Nepero

ESPONENZIALI
sp.exp(x)           e^x
sp.exp(sp.I*x)      e^(j x)

POTENZE
x**2                x^2
sp.sqrt(x)          radice

TRIGONOMETRIA
sp.sin(x)
sp.cos(x)
sp.tan(x)

NUMERI COMPLESSI
z = 3 + 2*sp.I

PARTE REALE / IMMAGINARIA
sp.re(z)
sp.im(z)

MODULO
sp.Abs(z)

CONIUGATO
sp.conjugate(z)

SEMPLIFICAZIONE
sp.simplify(expr)

ESPANSIONE
sp.expand(expr)

FATTORIZZAZIONE
sp.factor(expr)

RISCRIVERE
expr.rewrite(sp.exp)
expr.rewrite(sp.sin)

SOSTITUZIONE
expr.subs(t, 1)
expr.subs({t:1, k:2})

VALORE NUMERICO
expr.evalf()

STAMPA
print(expr)

ESEMPIO
Formula di Eulero:
    e^(jt) = cos(t) + j*sin(t)

    t, k = sp.symbols('t k', real=True)
    esp1 = sp.exp(sp.I*t)
    esp2 = sp.cos(t) + sp.I * sp.sin(t)
"""

# Se fossero necessarie altre funzioni o simboli, possono essere
# definiti nella sezione sopra "SIMBOLI E FUNZIONI"

esp1 = A*(T/2)*(sinc(f, T/2)**2)*(sp.exp(sp.I*sp.pi*f*T) + sp.exp(-sp.I*sp.pi*f*T))
esp2 = 2*A*T*(sinc(f, T)**2) - A*T*(sinc(f, T/2)**2)
esp3 = (3/2)*A*T*sinc(f, T/2)*sinc(f, T*(3/2)) - (A/2)*T*sinc(f, T/2)*sinc(f, T/2)

esp1 = A*2*T*sinc(f, 2*T) - 2*A*T*(sinc(f, T)**2) + A*(T/2)*(sinc(f, T/2)**2)
esp2 = 2*A*T*sinc(f, 2*T) - (3/2)*A*T*sinc(f, T/2)*sinc(f, (3/2)*T)

esp1 = (T/2)*(sinc(f, T/2)**2)*sp.exp(2*sp.pi*sp.I*f*(T/2)) - (T/2)*(sinc(f, T/2)**2)*sp.exp(-2*sp.pi*sp.I*f*(T/2))
esp2 = (2*sinc(f, T/2)*sp.cos(sp.pi*f*(3/2)*T) - 2*sinc(f, T)) / (sp.I*2*sp.pi*f)


if equal(esp1, esp2):
    print("Le due epressioni sono uguali")
else:
    print("Le due epressioni sono diverse")
