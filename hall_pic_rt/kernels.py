"""Najgorętsze kawałki kroku, przepisane na jedno przejście po czastkach.

Koszt zchodzi glownie na trzy rzeczy: rozłożenie czastek na węzły siatki,
odczytanie pola w ich miejscu i samo ich przesuwanie. 
Każda z nich jest prosta, ale liczona na całych tablicach naraz
tworzy kilkadziesiąt tablic pośrednich :(, a każda z nich raz jest zapisywana i raz
odczytywana z pamięci, wiec robi się sporo operacji.
"""

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:                                    # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        """Zastępnik, gdy tłumaczenie na kod maszynowy jest niedostępne."""
        def wrap(f):
            return f
        if args and callable(args[0]):
            return args[0]
        return wrap


@njit(cache=True, fastmath=False)
def deposit_cic(x1, x2, w, scale, h_r, h_th, N_r, N_th, out):
    """Rozkłada wielkość niesioną przez czastki na cztery najbliższe węzły.

    Węzły ponumerowane są tak samo jak w tablicy spłaszczonej wierszami, a
    kierunek wokół komory zawija się sam w sobie. Czastka, która zdążyła wyjść
    poza obszar w poprzek komory, liczona jest z udziałem wykraczającym poza
    przedział od zera do jedynki, dokładnie tak samo jak w wersji na tablicach,
    żeby obie drogi liczenia dawały ten sam wynik.
    """
    n = x1.shape[0]
    for p in range(n):
        f1 = x1[p] / h_r
        i = int(np.floor(f1))
        if i < 0:
            i = 0
        elif i > N_r - 1:
            i = N_r - 1
        a = f1 - i

        f2 = x2[p] / h_th
        jf = int(np.floor(f2))
        b = f2 - jf
        j = jf % N_th
        if j < 0:
            j += N_th
        j2 = j + 1
        if j2 == N_th:
            j2 = 0

        v = w[p] * scale
        base = i * N_th
        nxt = base + N_th
        va = v * a
        vna = v - va
        out[base + j] += vna * (1.0 - b)
        out[nxt + j] += va * (1.0 - b)
        out[base + j2] += vna * b
        out[nxt + j2] += va * b


@njit(cache=True, fastmath=False)
def push_species(x1, x2, x3, v1, v2, v3, w,
                 F_mean, F_fluc, F_th, use_fluc,
                 E_z, h_r, h_th, N_r, N_th,
                 dt, charge, mass, e_charge, b1, axial_on, L_th,
                 W_ax, W_sh, W_tr, W_tth, tracked, sums):
    """Odczytuje pole, przesuwa czastki i rozdziela pracę pola na składniki.

    Jedno przejście po czastkach: wyznacza udziały czterech najbliższych węzłów,
     odczytuje w nich obie składowe pola rozłożone na warstwę przyścienną i
    zaburzenia, wykonuje pełny krok schematu Borisa i wylicza pracę każdej składowej pola z osobna.

    Rozdział pracy jest ścisły. Zmiana energii czastki w kroku jest liniowa
    względem pola poza jednym członem kwadratowym, a ten jest rodzielany
    na skladowe iloczynem czyli kawaleczki sumuja sie do zmiany energi dE

    Sumy ważone wagą czastek zbieramy od razu w tablicy `sums`, żeby księga
    energii nie musiała potem jeszcze raz przechodzić po wszystkich czastkach.
    """
    n = x1.shape[0]
    qm = charge / mass
    half = 0.5 * dt * qm
    c_lin = 0.5 * charge * dt / e_charge
    c_quad = charge * charge * dt * dt / (4.0 * mass * e_charge)
    t1 = half * b1
    s1 = 2.0 * t1 / (1.0 + t1 * t1)

    s_ax = 0.0
    s_sh = 0.0
    s_tr = 0.0
    s_tth = 0.0

    for p in range(n):
        # ---- udziały czterech najbliższych węzłów ----
        f1 = x1[p] / h_r
        i = int(np.floor(f1))
        if i < 0:
            i = 0
        elif i > N_r - 1:
            i = N_r - 1
        a = f1 - i

        f2 = x2[p] / h_th
        jf = int(np.floor(f2))
        b = f2 - jf
        j = jf % N_th
        if j < 0:
            j += N_th
        j2 = j + 1
        if j2 == N_th:
            j2 = 0

        base = i * N_th
        nxt = base + N_th
        w00 = (1.0 - a) * (1.0 - b)
        w10 = a * (1.0 - b)
        w01 = (1.0 - a) * b
        w11 = a * b

        # ---- pole w miejscu czastki, rozłożone na składniki ----
        Em = (F_mean[base + j] * w00 + F_mean[nxt + j] * w10
              + F_mean[base + j2] * w01 + F_mean[nxt + j2] * w11)
        if use_fluc:
            Ef = (F_fluc[base + j] * w00 + F_fluc[nxt + j] * w10
                  + F_fluc[base + j2] * w01 + F_fluc[nxt + j2] * w11)
            Et = (F_th[base + j] * w00 + F_th[nxt + j] * w10
                  + F_th[base + j2] * w01 + F_th[nxt + j2] * w11)
        else:
            Ef = 0.0
            Et = 0.0

        E1 = Em + Ef
        E2 = Et
        E3 = E_z

        # ---- krok Borisa ----
        a1 = v1[p]
        a2 = v2[p]
        a3 = v3[p]
        vm1 = a1 + half * E1
        vm2 = a2 + half * E2
        vm3 = a3 + half * E3

        p2 = vm2 + vm3 * t1
        p3 = vm3 - vm2 * t1
        vp1 = vm1
        vp2 = vm2 + p3 * s1
        vp3 = vm3 - p2 * s1

        n1 = vp1 + half * E1
        n2 = vp2 + half * E2
        n3 = vp3 + half * E3
        v1[p] = n1
        v2[p] = n2
        v3[p] = n3
        x1[p] += n1 * dt
        # Kierunek wokół komory zawija się sam w sobie. Zawijamy od razu tutaj,
        # bo czastka jest już w rejestrach.
        u2 = x2[p] + n2 * dt
        u2 -= L_th * np.floor(u2 / L_th)
        x2[p] = u2
        if axial_on:
            x3[p] += n3 * dt

        # ---- praca każdej składowej pola z osobna ----
        k1 = c_lin * (a1 + vp1) + c_quad * E1
        k2 = c_lin * (a2 + vp2) + c_quad * E2
        k3 = c_lin * (a3 + vp3) + c_quad * E3

        d_ax = E3 * k3
        d_sh = Em * k1
        d_tr = Ef * k1
        d_tth = Et * k2

        wp = w[p]
        s_ax += wp * d_ax
        s_sh += wp * d_sh
        s_tr += wp * d_tr
        s_tth += wp * d_tth

        if tracked:
            W_ax[p] += d_ax
            W_sh[p] += d_sh
            W_tr[p] += d_tr
            W_tth[p] += d_tth

    sums[0] = s_ax
    sums[1] = s_sh
    sums[2] = s_tr
    sums[3] = s_tth


def warmup(dtype=np.float64):
    """Tłumaczy oba kernele na kod maszynowy, żeby nie działo się to w pomiarze."""
    if not HAVE_NUMBA:
        return
    z = np.zeros(1, dtype=dtype)
    o = np.zeros(4, dtype=dtype)
    g = np.zeros(4, dtype=dtype)
    deposit_cic(z, z, z, 1.0, 1.0, 1.0, 1, 1, g)
    push_species(z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.copy(),
                 g, g, g, True, 0.0, 1.0, 1.0, 1, 1, 1.0, -1.0, 1.0, 1.0, 0.0, True,
                 1.0, z.copy(), z.copy(), z.copy(), z.copy(), False, o)
