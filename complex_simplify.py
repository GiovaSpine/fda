import sympy as sp

# Data un espressione complessa, anche con parametro 'w', può
# essere utile semplificarla come a + jb
# Ad esempio in Nyquist o Bode per trovare i punti esatti

# Definizione simboli
w = sp.Symbol('w', real=True)
j = sp.I

# ---------------------------------------------------------

# Espressione complessa
expr = (9/10) * (1/(j*w)) * (1 - j*((w + sp.sqrt(5))/2)) * (1 - j*((w - sp.sqrt(5))/2))
expr = (9/10) * (1/(j*w)) * (1 / (1 + j*(w + sp.sqrt(5))/2)) * 1 / (1 + j*(w - sp.sqrt(5))/2)

# ---------------------------------------------------------

# Separa parte reale e immaginaria
expr_expanded = sp.simplify(expr)
re = sp.re(expr_expanded)
im = sp.im(expr_expanded)

re_simplified = sp.simplify(re)
im_simplified = sp.simplify(im)

print("Espressione originale:")
print(f"  G(jw) = {expr_expanded}")
print()
print("Forma  a + jb:")
print(f"  Re = {re_simplified}")
print(f"  Im = {im_simplified}")
print()
print(f"  G(jw) = ({re_simplified}) + j*({im_simplified})")