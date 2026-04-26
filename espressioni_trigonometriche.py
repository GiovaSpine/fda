import sympy as sp

# utile per comunicazioni numeriche
# se provi a risolvere un esericizio TCF in diversi modi
# otterrai espressioni diverse, che è complicato rendere uguali
# usando regole trigonometriche

f, A, T = sp.symbols('f A T', real=True)

def sinc(f, T):
    # espressione: sinc(fT) = sin(pi * f * T) / (pi * f * T)
    return sp.sin(sp.pi * f * T) / (sp.pi * f * T)

esp1 = A*(T/2)*(sinc(f, T/2)**2)*(sp.exp(sp.I*sp.pi*f*T) + sp.exp(-sp.I*sp.pi*f*T))
esp2 = 2*A*T*(sinc(f, T)**2) - A*T*(sinc(f, T/2)**2)
esp3 = (3/2)*A*T*sinc(f, T/2)*sinc(f, T*(3/2)) - (A/2)*T*sinc(f, T/2)*sinc(f, T/2)

esp1 = A*2*T*sinc(f, 2*T) - 2*A*T*(sinc(f, T)**2) + A*(T/2)*(sinc(f, T/2)**2)
esp2 = 2*A*T*sinc(f, 2*T) - (3/2)*A*T*sinc(f, T/2)*sinc(f, (3/2)*T)

esp1 = (T/2)*(sinc(f, T/2)**2)*sp.exp(2*sp.pi*sp.I*f*(T/2)) - (T/2)*(sinc(f, T/2)**2)*sp.exp(-2*sp.pi*sp.I*f*(T/2))
esp2 = (2*sinc(f, T/2)*sp.cos(sp.pi*f*(3/2)*T) - 2*sinc(f, T)) / (sp.I*2*sp.pi*f)

# riscrivo esp1 in termini di cos
esp1 = esp1.rewrite(sp.cos)
esp2 = esp2.rewrite(sp.cos)

# semplifico tutto
esp1_simpl = sp.trigsimp(sp.simplify(esp1))
esp2_simpl = sp.trigsimp(sp.simplify(esp2))

print("Esp1 semplificata:", esp1_simpl)
print("Esp2 semplificata:", esp2_simpl)

# confronto
diff = sp.simplify(esp1_simpl - esp2_simpl)
print("Differenza semplificata:", diff)
print("Uguali?", diff == 0)


# ===================================================================

'''
# così non funziona subito

result1 = sp.simplify(esp1 - esp2)
result2 = sp.simplify(sp.expand(esp1 - esp2))
result3 = sp.simplify(esp1.equals(esp2))

# 0 == le espressioni sono uguali
# True == le espressioni sono uguali
print(result1)
print(result2)
print(result3)
'''


'''
# ===================== SYMPY CHEAT SHEET =====================

# IMPORT
import sympy as sp

# VARIABILI
x, t, k = sp.symbols('x t k', real=True)   # real=True aiuta nelle semplificazioni

# COSTANTI IMPORTANTI
sp.I        # unità immaginaria (NON usare "j" o "i")
sp.pi       # pi greco
sp.E        # numero di Nepero

# ESPONENZIALI
sp.exp(x)                   # e^x
sp.exp(sp.I*x)              # e^(j x)

# POTENZE
x**2                        # x^2 (NON usare ^)
sp.sqrt(x)                  # radice

# TRIGONOMETRIA
sp.sin(x)
sp.cos(x)
sp.tan(x)

# NUMERI COMPLESSI
z = 3 + 2*sp.I

# PARTE REALE / IMMAGINARIA
sp.re(z)
sp.im(z)

# MODULO
sp.Abs(z)

# CONIUGATO
sp.conjugate(z)

# ESPRESSIONI (ESEMPIO)
expr = sp.exp(-t*(1 + sp.I*k))

# SEMPLIFICAZIONE
sp.simplify(expr)

# ESPANSIONE
sp.expand(expr)

# FATTORIZZAZIONE
sp.factor(expr)

# RISCRIVERE (utile per trig/esponenziali)
expr.rewrite(sp.exp)
expr.rewrite(sp.sin)

# SOSTITUZIONE
expr.subs(t, 1)
expr.subs({t:1, k:2})

# VALORE NUMERICO
expr.evalf()

# VERIFICARE UGUAGLIANZA (METODO MIGLIORE)
sp.simplify(expr1 - expr2)   # deve dare 0

# VERIFICA DIRETTA
expr1.equals(expr2)          # True / False / None

# STAMPA (IMPORTANTE!)
print(expr)

# =============================================================
'''

'''

prima roba
t, k = sp.symbols('t k', real=True)
esp1 = sp.exp(-t*(1 + sp.I*k))
esp2 = sp.exp(-t) * sp.exp(-sp.I*k*t)


Formula di Eulero
t, k = sp.symbols('t k', real=True)
e^(jt) = cos(t) + j*sin(t)
esp1 = sp.exp(sp.I*t)
esp2 = sp.cos(t) + sp.I * sp.sin(t)


# 10) coniugato
# MAT: (e^{jθ})* = e^{-jθ}
expr = sp.conjugate(sp.exp(sp.I*theta))
print(expr)   # → exp(-I*theta)


# 11) separare parte reale e immaginaria
# MAT: e^{jθ} = cos(θ) + j sin(θ)
expr = sp.exp(sp.I*theta)
print(sp.re(expr))   # cos(theta)
print(sp.im(expr))   # sin(theta)


# 12) verifica uguaglianza generale
# MAT: A = B ?
A = sp.exp(-t*(1 + sp.I*k))
B = sp.exp(-t)*sp.exp(-sp.I*k*t)
print(sp.simplify(A - B))   # → 0
'''