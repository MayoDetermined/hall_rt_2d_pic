"""Zderzenia kulombowskie między elektronami.

Rozgrywamy je sposobem par ze w każdej komórce łączymy elektrony losowo w pary i każdej parze
nadajemy jedno odchylenie prędkości względnej, o wielkości losowanej z rozkładu,
którego szerokość odpowiada sumie wszystkich drobnych odchyleń w ciągu jednego
kroku. Odchylenie obraca prędkość względną pary, nie zmieniając jej długości,
więc pęd i energia pary zostają dokładnie zachowane.

To ma znaczenie sporawe dla pytania. Zderzenia między elektronami wygladzaja
rozklad prędkości i zawracają energię z ogona do tla. Pominięcie ich mogłoby
zawyżać nadwyżkę szybkich elektronów, dlatego trzeba je mieć i chcemy umieć je
wyłączyć, żeby porównać jedno z drugim.

Częstość takich zderzeń maleje z energią jak jej potęga trzech drugich, więc
elektrony w ogonie zderzają się znacznie rzadziej niż tło. Ten sam moduł liczy
więc coś, co silnie dotyczy tła, a słabo samego ogona, i właśnie tę różnicę
warto zobaczyć w wyniku.
"""

import numpy as np

from hall_pic_rt.constants import E_CHARGE, M_ELECTRON, EPS0
from .pusher import cell_index


def coulomb_log(n_e, T_eV, cfg):
    """Zwraca logarytm Coulomba dla podanej gęstości i temperatury.

    Jest to logarytm z liczby czastek w kuli o promieniu długości ekranowania.
    Dla bardzo rzadkiej albo bardzo zimnej plazmy wzór przestaje mieć sens, więc
    wynik przycinamy do rozsądnego przedziału.
    """
    n = np.maximum(n_e, 1.0)
    T = np.maximum(T_eV, 1e-3)
    lam_D = np.sqrt(EPS0 * T / (E_CHARGE * n))
    lnL = np.log(12.0 * np.pi * n * lam_D**3)
    return np.clip(lnL, cfg.coulomb_lnL_min, cfg.coulomb_lnL_max)


def _pair_within_cells(cell, n_cells, rng):
    """Łączy czastki w losowe pary wewnątrz każdej komórki.

    Zwraca dwie tablice numerów czastek o równej długości; te same miejsca w
    obu tablicach tworzą parę. Gdy w komórce jest nieparzyście wiele czastek,
    jedna z nich w tym kroku nie bierze udziału, a że wybór pary jest losowany
    od nowa w każdym kroku, nie tworzy to obciążenia wyniku.
    """
    N = cell.size
    if N < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    # Najpierw tasujemy wszystkie czastki, potem układamy je po komórkach
    # porządkowaniem, które nie zmienia kolejności czastek z tej samej komórki.
    # Wychodzi na to samo co losowanie par w każdej komórce z osobna, ale liczy
    # się wyraźnie szybciej, bo porządkowanie po liczbach całkowitych z małego
    # zakresu jest tanie.
    perm = rng.permutation(N)
    order = perm[np.argsort(cell[perm], kind="stable")]

    sc = cell[order]

    counts = np.bincount(sc, minlength=n_cells)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))

    rank = np.arange(N) - starts[sc]

    n_pairs_cell = counts // 2
    take = rank < 2 * n_pairs_cell[sc]

    if not np.any(take):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    
    sel = order[take]
    slot = rank[take] % 2

    return sel[slot == 0], sel[slot == 1]


def _scatter_relative(ux, uy, uz, u, delta, rng):
    """Obraca prędkość względną pary o kąt wynikający z wylosowanego odchylenia.

    Odchylenie podaje tangens połowy kąta obrotu. Kierunek obrotu wokół
    prędkości względnej losujemy równomiernie. Długość prędkości względnej się
    nie zmienia, więc energia pary zostaje zachowana dokładnie.
    """
    d2 = delta * delta
    sin_t = 2.0 * delta / (1.0 + d2)
    one_m_cos = 2.0 * d2 / (1.0 + d2)
    phi = rng.random(delta.size) * 2.0 * np.pi
    cph, sph = np.cos(phi), np.sin(phi)

    u_perp = np.sqrt(ux * ux + uy * uy)
    safe = u_perp > 1e-10 * np.maximum(u, 1e-30)
    up = np.where(safe, u_perp, 1.0)

    dux = (ux / up) * uz * sin_t * cph - (uy / up) * u * sin_t * sph - ux * one_m_cos
    duy = (uy / up) * uz * sin_t * cph + (ux / up) * u * sin_t * sph - uy * one_m_cos
    duz = -u_perp * sin_t * cph - uz * one_m_cos

    # Gdy prędkość względna leży prawie wzdłuż trzeciej osi, powyższy wzór się
    # wykłada, więc dla takich par obracamy wprost względem tej osi.
    if not np.all(safe):
        m = ~safe
        s = np.sign(uz[m])
        s[s == 0] = 1.0
        dux[m] = u[m] * sin_t[m] * cph[m]
        duy[m] = u[m] * sin_t[m] * sph[m]
        duz[m] = -uz[m] * one_m_cos[m] + 0.0 * s
    return dux, duy, duz


def collide_ee(electrons, cfg, rng, dt_coll=None, ledger=True):
    """Rozgrywa zderzenia kulombowskie elektron-elektron w całym obszarze.

    Zwraca łączną zmianę energii elektronów wywołaną tymi zderzeniami. Powinna
    być bliska zeru, bo zderzenia jedynie zonglują energią między
    czastkami
    """
    N = electrons.N
    if N < 2:
        return 0.0
    dt_coll = cfg.dt if dt_coll is None else dt_coll

    cell = cell_index(electrons, cfg)
    i1, i2 = _pair_within_cells(cell, cfg.n_cells, rng)
    if i1.size == 0:
        return 0.0

    # Gęstość i temperatura elektronów w komórce, potrzebne do wielkości odchylenia.
    vol = cfg.h_r * cfg.h_th
    w_cell = np.bincount(cell, weights=electrons.aw, minlength=cfg.n_cells)
    n_cell = w_cell / vol
    eps_all = electrons.kinetic_energy_eV()

    e_cell = np.bincount(cell, weights=electrons.aw * eps_all, minlength=cfg.n_cells)
    T_cell = (2.0 / 3.0) * e_cell / np.maximum(w_cell, 1e-300)

    lnL_cell = coulomb_log(n_cell, T_cell, cfg)

    c1 = cell[i1]
    n_L = n_cell[c1]
    lnL = lnL_cell[c1]

    v1x, v1y, v1z = electrons.v1[i1], electrons.v2[i1], electrons.v3[i1]
    v2x, v2y, v2z = electrons.v1[i2], electrons.v2[i2], electrons.v3[i2]

    ux, uy, uz = v1x - v2x, v1y - v2y, v1z - v2z
    u = np.sqrt(ux * ux + uy * uy + uz * uz)
    u = np.maximum(u, 1e-6)          # para o zerowej prędkości względnej nic nie odczuwa

    # Wariancja odchylenia dla pary czastek tego samego gatunku. Masa
    # zredukowana wynosi wtedy połowę masy elektronu.
    m_r = 0.5 * M_ELECTRON

    var = (E_CHARGE**4 * n_L * lnL * dt_coll) / (
        8.0 * np.pi * EPS0**2 * m_r**2 * u**3)

    delta = rng.normal(0.0, 1.0, u.size) * np.sqrt(var)

    # Gdy odchylenie robi się duże, przybliżenie wieloma drobnymi uderzeniami
    # przestaje obowiązywać i rozproszenie jest po prostu równomierne we
    # wszystkie strony. Odpowiada temu losowanie cosinusa kąta po całym zakresie.
    strong = var > 1.0
    if np.any(strong):
        cos_t = 2.0 * rng.random(int(np.count_nonzero(strong))) - 1.0
        # tangens połowy kąta z wylosowanego cosinusa
        delta[strong] = np.sqrt(np.maximum((1.0 - cos_t) / (1.0 + cos_t), 0.0))

    dux, duy, duz = _scatter_relative(ux, uy, uz, u, delta, rng)

    # Poprawka na różne wagi czastek. Czastka o mniejszej wadze odchyla się
    # zawsze, a o większej z prawdopodobieństwem równym stosunkowi wag. Przy
    # jednakowych wagach obie odchylają się zawsze.
    w1 = electrons.w[i1]
    w2 = electrons.w[i2]
    ratio = np.minimum(w1, w2) / np.maximum(np.maximum(w1, w2), 1e-300)
    draw = rng.random(i1.size)

    act1 = (w1 <= w2) | (draw < ratio)
    act2 = (w2 <= w1) | (draw < ratio)

    if ledger and electrons.tracked:
        ke1_before = 0.5 * M_ELECTRON * (v1x**2 + v1y**2 + v1z**2) / E_CHARGE
        ke2_before = 0.5 * M_ELECTRON * (v2x**2 + v2y**2 + v2z**2) / E_CHARGE

    h = 0.5
    a1 = act1.astype(float) * h
    a2 = act2.astype(float) * h

    electrons.v1[i1] = v1x + a1 * dux
    electrons.v2[i1] = v1y + a1 * duy
    electrons.v3[i1] = v1z + a1 * duz
    electrons.v1[i2] = v2x - a2 * dux
    electrons.v2[i2] = v2y - a2 * duy
    electrons.v3[i2] = v2z - a2 * duz

    if ledger and electrons.tracked:
        ke1_after = 0.5 * M_ELECTRON * (
            electrons.v1[i1]**2 + electrons.v2[i1]**2 + electrons.v3[i1]**2) / E_CHARGE
        
        ke2_after = 0.5 * M_ELECTRON * (
            electrons.v1[i2]**2 + electrons.v2[i2]**2 + electrons.v3[i2]**2) / E_CHARGE
        electrons.W_ee[i1] += ke1_after - ke1_before
        
        electrons.W_ee[i2] += ke2_after - ke2_before
        dE = float(np.sum(w1 * (ke1_after - ke1_before))
                   + np.sum(w2 * (ke2_after - ke2_before)))
        
        return dE

    return 0.0
