"""Przenoszenie informacji między siatką a czastkami oraz ruch czastek.
"""

import numpy as np

from hall_pic_rt.constants import E_CHARGE
from . import kernels


def charge_density(sp, cfg, out=None):
    """Rozkłada ładunek jednego gatunku na węzły, jako tablica spłaszczona.

    Zwraca gęstość liczoną na pełną komórkę, bez poprawki na węzły przy
    ściankach. Poprawke naklada ten, kto składa gęstości obu gatunków, żeby
    zrobić to raz dka optymalizacji.
    """
    n_total = cfg.n_r_nodes * cfg.n_th_nodes
    if out is None:
        out = np.zeros(n_total)

    if sp.N == 0:
        return out
    
    scale = sp.charge / (cfg.h_r * cfg.h_th)
    
    if kernels.HAVE_NUMBA:
        kernels.deposit_cic(sp.ax1, sp.ax2, sp.aw, scale, cfg.h_r, cfg.h_th,
                            cfg.N_r, cfg.n_th_nodes, out)
        return out
    
    idxs, wts = cic_weights(sp, cfg)
    out += _accumulate(idxs, wts, sp.aw * scale, n_total)
    
    return out


def add_charge_of_new(sp, cfg, start, out, sign=1.0):
    """Poprawia gotowy rozkład ładunku o czastki dopisane po `start`.

    Ujemny znak służy do odjęcia czastek, które zaraz znikną. Dzięki temu
    rozkład ładunku gatunku pchanego rzadziej daje się utrzymywać w zgodzie z
    jego tablicami bez przeliczania całości w każdym kroku.
    """
    if sp.N <= start:
        return out
    scale = sign * sp.charge / (cfg.h_r * cfg.h_th)
    
    w = sp.w[start:sp.N]
    
    x1 = sp.x1[start:sp.N]
    x2 = sp.x2[start:sp.N]
    
    if kernels.HAVE_NUMBA:
        kernels.deposit_cic(x1, x2, w, scale, cfg.h_r, cfg.h_th,
                            cfg.N_r, cfg.n_th_nodes, out)
        return out
    
    n_total = cfg.n_r_nodes * cfg.n_th_nodes
    idxs, wts = _cic_from_positions(x1, x2, cfg)
    
    out += _accumulate(idxs, wts, w * scale, n_total)
    
    return out


def add_charge_of_mask(sp, cfg, mask, out, sign=1.0):
    """Poprawia gotowy rozkład ładunku o zaznaczone czastki."""
    idx = np.flatnonzero(mask)
    
    if idx.size == 0:
        return out
    
    scale = sign * sp.charge / (cfg.h_r * cfg.h_th)
    
    x1 = sp.x1[idx]
    x2 = sp.x2[idx]
    w = sp.w[idx]
    
    if kernels.HAVE_NUMBA:
        kernels.deposit_cic(x1, x2, w, scale, cfg.h_r, cfg.h_th,
                            cfg.N_r, cfg.n_th_nodes, out)
        return out
    
    n_total = cfg.n_r_nodes * cfg.n_th_nodes
    idxs, wts = _cic_from_positions(x1, x2, cfg)
    
    out += _accumulate(idxs, wts, w * scale, n_total)
    return out


def cell_index(sp, cfg):
    """Zwraca numer komórki, w której leży każda czastka."""
    i = np.floor(sp.ax1 / cfg.h_r).astype(np.int64)
    
    np.clip(i, 0, cfg.N_r - 1, out=i)
    j = np.mod(np.floor(sp.ax2 / cfg.h_th).astype(np.int64), cfg.N_th)
    
    return i * cfg.N_th + j


def cic_weights(sp, cfg):
    """Zwraca cztery najbliższe węzły każdej czastki wraz z ich udziałami.

    Ten sam wynik przydaje się dwa razy na step a) przy rozkładaniu ładunku i b) przy
    odczycie pola. Oba dzieją się przed przesunięciem czastek wiec mozna to zoptymalizowac.
    """
    return _cic_from_positions(sp.ax1, sp.ax2, cfg)


def _cic_from_positions(x1, x2, cfg):
    """To samo, ale dla podanych wprost położeń."""
    f1 = x1 / cfg.h_r
    i = np.floor(f1).astype(np.int64)
    np.clip(i, 0, cfg.N_r - 1, out=i)
    a = f1 - i

    f2 = x2 / cfg.h_th
    j = np.floor(f2).astype(np.int64)
    b = f2 - j
    j = np.mod(j, cfg.N_th)
    j2 = np.mod(j + 1, cfg.N_th)

    n2 = cfg.n_th_nodes
    idx00 = i * n2 + j
    idx10 = (i + 1) * n2 + j
    idx01 = i * n2 + j2
    idx11 = (i + 1) * n2 + j2
    
    w00 = (1.0 - a) * (1.0 - b)
    w10 = a * (1.0 - b)
    w01 = (1.0 - a) * b
    w11 = a * b
    
    return (idx00, idx10, idx01, idx11), (w00, w10, w01, w11)


def _accumulate(idxs, wts, values, n_total):
    """Sumuje na węzłach udziały czastek rozłożone na cztery krance."""
    out = np.zeros(n_total)
    
    for idx, wt in zip(idxs, wts):
        out += np.bincount(idx, weights=values * wt, minlength=n_total)
    return out


def _fix_wall_nodes(arr):
    """Podwaja wartości w węzłach na ściankach, bo obejmują tylko pół komórki."""
    arr[0, :] *= 2.0
    arr[-1, :] *= 2.0


def deposit_charge(species_list, cfg, cached=None):
    """Rozkłada ładunek wszystkich gatunków na węzły siatki."""
    n_total = cfg.n_r_nodes * cfg.n_th_nodes
    rho_flat = np.zeros(n_total)
    cell_vol = cfg.h_r * cfg.h_th
    
    for k, sp in enumerate(species_list):
        if sp.N == 0:
            continue
        idxs, wts = cached[k] if cached is not None else cic_weights(sp, cfg)
        rho_flat += _accumulate(idxs, wts, sp.charge * sp.aw / cell_vol, n_total)
    
    rho = rho_flat.reshape(cfg.n_r_nodes, cfg.n_th_nodes)
    _fix_wall_nodes(rho)
    return rho


def deposit_number_density(sp, cfg, cached=None):
    """Rozkłada na węzły samą liczebność jednego gatunku."""
    n_total = cfg.n_r_nodes * cfg.n_th_nodes
    
    if sp.N == 0:
        return np.zeros((cfg.n_r_nodes, cfg.n_th_nodes))
    idxs, wts = cached if cached is not None else cic_weights(sp, cfg)
    
    n = _accumulate(idxs, wts, sp.aw / (cfg.h_r * cfg.h_th), n_total)
    n = n.reshape(cfg.n_r_nodes, cfg.n_th_nodes)
    
    _fix_wall_nodes(n)
    return n


def deposit_moment(sp, cfg, values, cached=None):
    """Rozkłada na węzły dowolną wielkość niesioną przez czastki, ważoną ich wagą."""
    n_total = cfg.n_r_nodes * cfg.n_th_nodes
    
    if sp.N == 0:
        return np.zeros((cfg.n_r_nodes, cfg.n_th_nodes))
    idxs, wts = cached if cached is not None else cic_weights(sp, cfg)
    
    m = _accumulate(idxs, wts, sp.aw * values, n_total)
    m = m.reshape(cfg.n_r_nodes, cfg.n_th_nodes)
    
    _fix_wall_nodes(m)
    return m


def gather(field, idxs, wts, N):
    """Odczytuje wartość pola w miejscu każdej czastki."""
    f = field.ravel()
    out = np.zeros(N)
    
    for idx, wt in zip(idxs, wts):
        out += f[idx] * wt
    return out


def boris_push(sp, E_parts, cfg, ledger_out=None, dt=None):
    """Przesuwa czastki o jeden krok i rozdziela pracę pola na składniki.

    Pole podajemy rozłożone na składniki, każdy jako trójka wartości w miejscu
    czastki. Czastka odczuwa ich sumę, ale rachunek prowadze osobno dla
    każdego, bo to wlasnie chce zbadac.
    """
    N = sp.N
    if N == 0:
        return
    dt = cfg.dt if dt is None else dt
    qm = sp.charge / sp.mass
    half = 0.5 * dt * qm

    # Pole pełne to suma składowych. Składowe nieobecne podajemy jako brak, a
    # nie jako zera, żeby nie liczyć mnożeń przez zero na całych tablicach.
    
    E1 = np.zeros(N)
    E2 = np.zeros(N)
    E3 = np.zeros(N)
    
    for (e1, e2, e3) in E_parts.values():
        if e1 is not None:
            E1 += e1
        if e2 is not None:
            E2 += e2
        if e3 is not None:
            E3 += e3

    v1 = sp.av1.copy()
    v2 = sp.av2.copy()
    v3 = sp.av3.copy()

    # Pierwsza połowa rozpędzania.
    vm1 = v1 + half * E1
    vm2 = v2 + half * E2
    vm3 = v3 + half * E3

    # Obrót w polu magnetycznym, które leży w poprzek komory.
    b1, b2, b3 = cfg.B_vector(sp.ax1)
    
    t1 = half * b1
    t2 = half * b2
    t3 = half * b3
    
    tsq = t1*t1 + t2*t2 + t3*t3
    s = 2.0 / (1.0 + tsq)
    s1, s2, s3 = s*t1, s*t2, s*t3

    if np.all(t2 == 0.0) and np.all(t3 == 0.0):
        # Pole magnetyczne leży dokładnie wzdłuż pierwszej osi, więc obrót
        # dotyczy tylko dwóch pozostałych składowych prędkości. Z czego korzystam
        
        p2 = vm2 + vm3 * t1
        p3 = vm3 - vm2 * t1
        
        vp1 = vm1
        vp2 = vm2 + p3 * s1
        vp3 = vm3 - p2 * s1
    else:
        p1 = vm1 + (vm2*t3 - vm3*t2)
        p2 = vm2 + (vm3*t1 - vm1*t3)
        p3 = vm3 + (vm1*t2 - vm2*t1)
        
        vp1 = vm1 + (p2*s3 - p3*s2)
        vp2 = vm2 + (p3*s1 - p1*s3)
        vp3 = vm3 + (p1*s2 - p2*s1)

    # Druga połowa rozpędzania i przesunięcie po płaszczyźnie.
    sp.v1[:N] = vp1 + half * E1
    sp.v2[:N] = vp2 + half * E2
    sp.v3[:N] = vp3 + half * E3

    sp.x1[:N] = sp.ax1 + sp.v1[:N] * dt
    sp.x2[:N] = sp.ax2 + sp.v2[:N] * dt

    if cfg.axial_length > 0.0:
        # Położenie wzdłuż osi komory przesuwamy tak samo jak pozostałe, choć
        # nie ma go na siatce. Służy tylko do rozstrzygnięcia, kiedy czastka
        # opuści obszar wzdłuż osi.
        sp.x3[:N] = sp.ax3 + sp.v3[:N] * dt

    if ledger_out is None:
        return

    # Rozdział pracy pola na składniki, w elektronowoltach na czastkę.
    c_lin = 0.5 * sp.charge * dt / E_CHARGE
    c_quad = sp.charge**2 * dt**2 / (4.0 * sp.mass * E_CHARGE)
    
    # Dla każdej osi mnożnik jest ten sam dla wszystkich składowych pola, więc
    # liczymy go raz, a potem każda składowa to już jedno mnożenie.
    k1 = c_lin * (v1 + vp1) + c_quad * E1
    k2 = c_lin * (v2 + vp2) + c_quad * E2
    k3 = c_lin * (v3 + vp3) + c_quad * E3
    
    for name, (e1, e2, e3) in E_parts.items():
        acc = None
        for e, k in ((e1, k1), (e2, k2), (e3, k3)):
            if e is None:
                continue
            term = e * k
            acc = term if acc is None else acc + term
        ledger_out[name] = 0.0 if acc is None else acc
