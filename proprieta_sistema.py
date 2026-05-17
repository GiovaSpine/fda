import numpy as np


def __check_matrix(A, square=False):
    '''
    Semplice controllo della matrice A:
    deve essere una matrice quadrata di numeri reali.

    Parametri:
        A (matrix)   : La matrice da controllare
        square (bool): Se la matrice deve essere quadrata o no
    '''
    if isinstance(A, np.ndarray):
        if A.ndim != 2:
            raise ValueError(f"La matrice deve essere 2D, ha {A.ndim} dimensioni")
        if not np.issubdtype(A.dtype, np.number):
            raise TypeError(f"La matrice deve contenere numeri, dtype è {A.dtype}")

    elif isinstance(A, list):
        if len(A) == 0:
            raise ValueError("A non può essere vuota")
        if not isinstance(A[0], list):
            raise TypeError("La matrice deve essere una lista di liste")
        m = len(A[0])
        for i, row in enumerate(A):
            if not isinstance(row, list):
                raise TypeError(f"A[{i}] deve essere una lista")
            if len(row) != m:
                raise ValueError(f"A[{i}] ha {len(row)} colonne, attese {m}")
            for j, val in enumerate(row):
                if not isinstance(val, (int, float, complex, np.number)):
                    raise TypeError(f"A[{i}][{j}] deve essere un numero")

    else:
        raise TypeError(f"La matrice deve essere una lista o numpy ndarray, è {type(A)}")
    
    if square:
        aux = np.array(A)
        if aux.shape[0] != aux.shape[1]:
            raise ValueError(f"La matrice deve essere una matrice quadrata, shape è {aux.shape}")
    
    if not np.isrealobj(np.array(A)):
        raise ValueError("La matrice deve essere una matrice di numeri reali")
    

def __trova_q(A, eigenvalue, m, tol=1e-9):
    '''
    Trova l'indice q (dimensione del blocco di Jordan più grande)
    per un dato autovalore di A.

    Parametri:
        A          : Matrice numpy (n x n)
        eigenvalue : Autovalore di cui cercare q
        m          : Molteplicità algebrica dell'autovalore
        tol        : Tolleranza per il calcolo del rango
    '''
    n = A.shape[0]
    M = A - eigenvalue * np.eye(n)   # (A - λI)

    Mk = np.eye(n)   # (A - λI)^k, parte da k=0

    for k in range(1, m + 1):
        Mk = Mk @ M                  # eleva alla potenza k
        rango    = np.linalg.matrix_rank(Mk, tol=tol)
        nullita  = n - rango

        # print(f"k={k}:  rango={(rango)}  nullità={nullita}")

        if nullita >= m:
            # print(f"\n→ q = {k}  (blocco di Jordan massimo {k}x{k})")
            return k

    return m   # fallback


def stabilita(A):
    '''
    Calcola i modi del sistema e la conseguente stabilità (interna).

    Parametri:
        A: Matrice di numeri reali che rapprenseta il sistema
    '''
    __check_matrix(A, square=True)

    A = np.array(A)

    # Per calcolare la stabilità ci servono gli autovalori
    autovalori = np.linalg.eigvals(A)

    # Una volta avuti gli autovalori conosciamo i modi associati

    # la molteplicità algebrica 'ma' si può contare
    unique, ma = np.unique(np.round(autovalori, 6), return_counts=True)

    # la molteplicità geometrica si può ottenere con Rouché-Capelli
    # si ha il sistema lineare A*v = lambda*v --> lambda*I - A = 0
    # mg = n - rango(lambda*I - A)
    n = A.shape[0]
    
    mg = []
    for val in unique:
        M = A - val * np.eye(n)
        rank = np.linalg.matrix_rank(M)
        mg.append(n - rank)
    
    # Il sistema è:
    # - Asintoticamente stabile se tutti i valori reali sono < 0
    # - Marginalmente stabile se tutti i valori reali sono <= 0
    #   e i valori pari a 0 hanno 'ma' = 'mg' --> q = 1 (q dim. blocco di Jordan)
    # - Instabile altrimenti

    def stampa_modo(val, ma, mg):
        print("modi associati: ", end='')
        polinomi = [""]
        if ma != mg:
            q = __trova_q(A, val, ma)
            print(f"(q = {q})  ", end='')
            for i in range(1, q):
                polinomi.append(f"t^{i}/{i}!")
        sigma = np.real(val)
        omega = abs(np.imag(val))   # Sempre positivo
        prefix = f"e^({sigma}t)" if sigma != 0 else ""
        if omega != 0:
            for p in polinomi:
                print(f"{prefix}cos({omega}t){p},  {prefix}sin({omega}t){p},", end='  ')
        else:
            for p in polinomi:
                if sigma != 0:
                    print(f"e^({sigma}t){p}, ", end='')
                else:
                    print(f"1{p}, ", end='')
        print()
    
    # Stampa gli autovalori
    k = 1
    stability = "asintoticamente stabile"
    for val, a, g in zip(unique, ma, mg):
        # ATTENZIONE
        # I complessi sono coniugati e quindi si stampano insieme
        if np.imag(val) < 0:
            continue  # Salta il coniugato negativo, ci pensa quello positivo
        
        sigma = np.round(np.real(val), 10)
        omega = np.round(np.imag(val), 10)
        
        if omega != 0:
            print(f"lambda{k} = {np.round(sigma,4)} ± {np.round(omega,4)}j   ma = {a}   mg = {g}")
        else:
            print(f"lambda{k} = {np.round(val.real,4)}   ma = {a}   mg = {g}")

        # Stampiamo il modo
        stampa_modo(val, a, g)

        # Otteniamo la stabilità
        if np.real(val) > 0:
            # A prescindere il modo associato è instabile
            stability = "instabile"
        elif np.real(val) == 0:
            # Il modo può essere marginalmente stabile o instabile
            # a seconda se ma = mg (q = 1) --> stabile
            if stability != "instabile":
                if a == g:
                    stability = "marginalmente stabile"
                else:
                    stability = "instabile" 

        print()
        k += 1
    
    print(f"--> Il sistema è {stability}\n") 
    
    
def raggiungibilita(A, B):
    '''
    Calcola se il sistema è raggiungibile in base alla matrice 'Mr':
    rango([B | AB | ... | A^(n-1)B]) = n

    Parametri:
        A: Matrice A n x n, di x' = Ax + Bu
        B: Matrice B n x r, di x' = Ax + Bu
    '''
    __check_matrix(A, square=True)
    __check_matrix(B)
    A = np.array(A)
    B = np.array(B)
    if A.shape[0] != B.shape[0]:
        raise ValueError("'B' deve avere 'n' righe")

    n = A.shape[0]

    Mr = B
    prev = B
    for _ in range(1, n):
        prod = np.dot(A, prev)
        Mr = np.append(Mr, prod, axis=1)
        prev = prod
    print("Mr =")
    print(np.array(Mr))
    print()

    rango = np.linalg.matrix_rank(Mr)
    if rango == n:
        print(f"rango(Mr) = {rango} == (n = {n})\n--> Il sistema è completamente raggiungibile")
    else:
        print(f"rango(Mr) = {rango} < (n = {n})\n--> Il sistema NON è completamente raggiungibile")
    print()

    
def osservabilita(A, C):
    '''
    Calcola se il sistema è osservabile in base alla matrice 'Mo':
    rango (  C     )  = n
          (  CA    )  
          (  ...   )
          (CA^(n-1))

    Parametri:
        A: Matrice A n x n, di x' = Ax + Bu
        C: Matrice C m x n di y = Cx + Du
    '''
    __check_matrix(A, square=True)
    __check_matrix(C)
    A = np.array(A)
    C = np.array(C)
    if A.shape[1] != C.shape[1]:
        raise ValueError("'C' deve avere 'n' colonne")

    n = A.shape[0]

    Mo = C
    prev = C
    for _ in range(1, n):
        prod = np.dot(prev, A)
        Mo = np.vstack([Mo, prod])
        prev = prod
    print("Mo =")
    print(np.array(Mo))
    print()

    rango = np.linalg.matrix_rank(Mo)
    if rango == n:
        print(f"rango(Mo) = {rango} == (n = {n})\n--> Il sistema è completamente osservabile")
    else:
        print(f"rango(Mo) = {rango} < (n = {n})\n--> Il sistema NON è completamente osservabile")
    print()


def raggiungibilita_PBH(A, B, stampa=True):
    '''
    Calcola se il sistema è raggiungibile usando il lemma PBH:
    per ogni lambda:
    rango[lambdaI - A | B] = n

    Parametri:
        A: Matrice A n x n, di x' = Ax + Bu
        B: Matrice B n x r, di x' = Ax + Bu
        stampa: Booleano per stampare i risultati o no
    '''
    __check_matrix(A, square=True)
    __check_matrix(B)
    A = np.array(A)
    B = np.array(B)
    if A.shape[0] != B.shape[0]:
        raise ValueError("'B' deve avere 'n' righe")

    # Dizionario nella forma (autovalore, {True, False})
    # che indica quali autovalori sono raggiungibili o no
    raggiungibili = {}

    n = A.shape[0]

    autovalori = list(set(np.linalg.eigvals(A)))
    
    for av in autovalori:
        # Ricaviamo la matrice [av*I - A | B]
        aux = (np.eye(n) * av) - A
        aux = np.append(aux, B, axis=1)
        if stampa:
            print(f"per lambda = {av}")
            print("[lambda*I - A | B] =")
            print(np.array(aux))
        rango = np.linalg.matrix_rank(aux)
        if rango < n:
            raggiungibili[av] = False
            if stampa:
                print(f"rango = {rango} < (n = {n})  B non riesce a far recuperare il rango massimo")
                print(f"--> Lo stato del modo associato a {av} è irraggiungibile")
        else:
            raggiungibili[av] = True
            if stampa:
                print(f"--> rango = {rango} == (n = {n})  B riesce a far recuperare il rango massimo")
                print(f"--> Lo stato del modo associato a {av} è raggiungibile")
        if stampa:
            print()

    return raggiungibili


def osservabilita_PBH(A, C, stampa=True):
    '''
    Calcola se il sistema è osservabile usando il lemma PBH:
    per ogni lambda:
    rango[lambdaI - A] = n
         [     C     ]

    Parametri:
        A: Matrice A n x n, di x' = Ax + Bu
        C: Matrice C m x n, di y = Cx + Du
        stampa: Booleano per stampare i risultati o no
    '''
    __check_matrix(A, square=True)
    __check_matrix(C)
    A = np.array(A)
    C = np.array(C)
    if A.shape[1] != C.shape[1]:
        raise ValueError("'C' deve avere 'n' colonne")
    
    # Dizionario nella forma (autovalore, {True, False})
    # che indica quali autovalori sono osservabili o no
    osservabili = {}

    n = A.shape[0]

    autovalori = list(set(np.linalg.eigvals(A)))
    
    for av in autovalori:
        # Ricaviamo la matrice [av*I - A]
        #                      [   C    ]
        aux = (np.eye(n) * av) - A
        aux = np.vstack([aux, C])
        if stampa:
            print(f"per lambda = {av}")
            print("[lambda*I - A] =")
            print("[     C      ]")
            print(np.array(aux))
        rango = np.linalg.matrix_rank(aux)
        if rango < n:
            osservabili[av] = False
            if stampa:
                print(f"rango = {rango} < (n = {n})  C non riesce a far recuperare il rango massimo")
                print(f"--> Lo stato del modo associato a {av} non è osservabile")
        else:
            osservabili[av] = True
            if stampa:
                print(f"--> rango = {rango} == (n = {n})  C riesce a far recuperare il rango massimo")
                print(f"--> Lo stato del modo associato a {av} è osservabile")
        if stampa:
            print()
    
    return osservabili


def stabilita_BIBO(A, B, C):
    '''
    Calcola la BIBO stabilità (stabilità esterna), mediante lo studio dei poli,
    che sono gli autovalori di A raggiungibili e osservabili.

    Parametri:
        A: Matrice A n x n
        B: Matrice B n x r, di x' = Ax + Bu
        C: Matrice C m x n, di y = Cx + Du
    '''
    __check_matrix(A, square=True)
    __check_matrix(B)
    __check_matrix(C)
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    if A.shape[0] != B.shape[0]:
        raise ValueError("'B' deve avere 'n' righe")
    if A.shape[1] != C.shape[1]:
        raise ValueError("'C' deve avere 'n' colonne")
    
    raggiungibili = raggiungibilita_PBH(A, B, False)
    osservabili = osservabilita_PBH(A, C, False)

    print("I poli del sistema sono gli autovalori raggiungibili e osservabili, ossia:")

    stabilita = "stabile"
    for (av, ragg), (_, oss) in zip(raggiungibili.items(), osservabili.items()):
        if ragg and oss:
            if np.real(av) > 0.0:
                stabilita = "instabile"
                print(f"- lambda = {av}  con parte reale > 0 (polo instabile)")
            else:
                print(f"- lambda = {av}")
    print(f"\nIl sistema è BIBO {stabilita}")
    print()

