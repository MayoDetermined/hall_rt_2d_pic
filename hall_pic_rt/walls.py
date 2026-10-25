"""Obsługa ścianek komory i kierunku zawijającego się wokół niej.

Ścianki stoją po obu stronach w poprzek komory. Kierunek wokół komory nie ma
brzegu, więc czastka, która z niego wyjdzie, wraca z drugiej strony.

Przy każdym zetknięciu ze ścianką zbieramy metryczkę czastki, która ginie. Dzięki
temu wiadomo nie tylko ile energii ucieka na ścianki, ale też skąd ta energia się
wzięła, zanim tam trafiła.
"""

import numpy as np

from hall_pic_rt.constants import E_CHARGE
from .species import LEDGER_FIELDS, ORIG_WALL


class WallSurface:
    """Ładunek zebrany na powierzchni obu ścianek, osobno wzdłuż obwodu.

    Trzymamy gęstość powierzchniową ładunku w tych samych miejscach na obwodzie,
    w których stoją węzły siatki, żeby dało się ją podać wprost do wyznaczania
    potencjału. Czastka, która ginie na ściance, zostawia na niej swój ładunek,
    a elektron wybity ze ścianki ten ładunek stamtąd zabiera; oba zdarzenia
    rozkładamy między dwa najbliższe miejsca tak samo, jak rozkłada się ładunek
    czastki na siatkę, więc ładunek nie ginie po drodze.
    """

    def __init__(self, cfg):
        """Zakłada obie powierzchnie nienaładowane."""
        self.cfg = cfg
        self.sigma = np.zeros((2, cfg.N_th))    # 0 to ścianka bliższa, 1 dalsza
        self.q_collected = 0.0                  # ładunek zebrany od początku przebiegu

    def reset(self):
        """Zeruje zebrany ładunek."""
        self.sigma[:] = 0.0
        self.q_collected = 0.0

    def collect(self, side, x2, q_w):
        """Dopisuje ładunek czastek, które zetknęły się ze ścianką.

        Podajemy, po której stronie i w którym miejscu na obwodzie czastka
        trafiła w ściankę oraz jaki ładunek ze sobą niosła. Ładunek podany jest
        na jednostkę długości osi komory, a powierzchnia jednego miejsca to
        szerokość komórki wzdłuż obwodu razy ta sama jednostka, więc gęstość
        powierzchniowa jest po prostu ilorazem tych dwóch.
        """
        if x2 is None or np.size(x2) == 0:
            return
        
        cfg = self.cfg
        
        f = x2 / cfg.h_th
        j = np.floor(f).astype(np.int64)
        b = f - j
        j = np.mod(j, cfg.N_th)
        j2 = np.mod(j + 1, cfg.N_th)
        val = q_w / cfg.h_th
        
        for s in (0, 1):
            m = side == s
            if not np.any(m):
                continue
            row = self.sigma[s]
            row += np.bincount(j[m], weights=(val[m] * (1.0 - b[m])), minlength=cfg.N_th)
            row += np.bincount(j2[m], weights=(val[m] * b[m]), minlength=cfg.N_th)
        
        self.q_collected += float(np.sum(q_w))

    def total_charge(self):
        """Cały ładunek zgromadzony na obu ściankach, na jednostkę długości osi."""
        return float(np.sum(self.sigma) * self.cfg.h_th)

    def measures(self, phi):
        """Zwraca miary opisujące stan obu ścianek, do diagnostyki.

        Poza średnim potencjałem podajemy jego zmienność wzdłuż obwodu.
        """
        lo, hi = phi[0], phi[-1]
        return {
            "wall_phi_lo": float(lo.mean()),
            "wall_phi_hi": float(hi.mean()),
            "wall_phi_rms": float(np.sqrt(0.5 * (lo.var() + hi.var()))),
            "wall_sigma_lo": float(self.sigma[0].mean()),
            "wall_sigma_hi": float(self.sigma[1].mean()),
            "wall_q": self.total_charge(),
        }


def see_yield(eps_eV, cfg):
    """Zwraca liczbę elektronów wtórnych przypadającą na jeden padający elektron.

    Wydajność rośnie liniowo z energią padającego elektronu i ma nasycenie
    wynikające z tego, że nadmiar elektronów wtórnych jest zawracany przez
    własny ładunek przestrzenny tuż przy ściance.
    """
    return np.minimum(cfg.see_gamma0 + cfg.see_slope * eps_eV, cfg.see_max)


def _emit_speeds(n, T_eV, rng):
    """Losuje prędkości elektronów wychodzących ze ścianki.

    Składowa prostopadła do ścianki pochodzi z rozkładu strumienia, a więc
    faworyzuje czastki szybsze, natomiast obie składowe wzdłuż ścianki są zwykłe
    cieplne. Tak wygląda rozkład czastek opuszczających powierzchnię.
    """
    from hall_pic_rt.constants import M_ELECTRON
    vth = np.sqrt(E_CHARGE * T_eV / M_ELECTRON)
    v_perp = vth * np.sqrt(-2.0 * np.log(np.maximum(rng.random(n), 1e-300)))
    v_a = rng.normal(0.0, vth, n)
    v_b = rng.normal(0.0, vth, n)
    return v_perp, v_a, v_b


# ---------------------------------------------------------------- ucieczka osiowa
def axial_events(sp, cfg, skip=None):
    """Wskazuje czastki, które opuściły obszar wzdłuż osi komory.
    """
    out = {"w": 0.0, "E_eV": 0.0, "ledger": {a: 0.0 for a in LEDGER_FIELDS},
           "E_birth": 0.0, "n": 0, "kill": None, "q_up": 0.0, "q_down": 0.0}
    
    if sp.N == 0 or cfg.axial_length <= 0.0:
        return out

    gone = (sp.ax3 < 0.0) | (sp.ax3 > cfg.axial_length)
    
    if skip is not None:
        gone &= ~skip
    
    if not np.any(gone):
        return out

    idx = np.nonzero(gone)[0]
    w = sp.w[idx]
    eps = 0.5 * sp.mass * (sp.v1[idx]**2 + sp.v2[idx]**2 + sp.v3[idx]**2) / E_CHARGE
    
    out["w"] = float(np.sum(w))
    out["E_eV"] = float(np.sum(w * eps))
    
    out["n"] = int(idx.size)
    out["kill"] = gone
    
    # Strona, którą czastka odeszła, decyduje o znaku prądu. Wyjscie na
    # stronę anody i wyjscie na strone wylotu daja prąd przeciwnych znakow, a
    # ich różnica jest prądem, który obwód widzi jako wyladowanie.
    
    up = sp.x3[idx] < 0.0
    out["q_up"] = sp.charge * float(np.sum(w[up]))
    out["q_down"] = sp.charge * float(np.sum(w[~up]))
    
    if sp.tracked:
        for a in LEDGER_FIELDS:
            out["ledger"][a] = float(np.sum(w * getattr(sp, a)[idx]))
        out["E_birth"] = float(np.sum(w * sp.E_birth[idx]))
    return out


def apply_axial_loss(sp, cfg):
    """Jak wyżej, ale od razu usuwa czastki. Zostawione dla prostych zastosowań."""
    out = axial_events(sp, cfg)
    if out["kill"] is not None:
        sp.remove_mask(out["kill"], stable=getattr(cfg, "stable_compact", False))
    return out


# ---------------------------------------------------------------- ścianki
def wall_events(sp, cfg, rng, t_now, allow_see=True, wrapped=False):
    """Zawija kierunek wokół komory i rozstrzyga zetknięcia ze ściankami.

    Zwraca opis tego, co stało się na ściankach: ile wagi i energii ubyło po
    każdej stronie, ile ładunku zostało na powierzchni, jakie elektrony wtórne
    mają zostać wybite oraz jaka była metryczka czastek, które zginęły.
    """
    out = {
        "w_lo": 0.0, "w_hi": 0.0,
        "E_out_eV": 0.0, "E_see_eV": 0.0,
        "w_see": 0.0, "n_see": 0,
        "q_wall": 0.0,
        "ledger_lost": {a: 0.0 for a in LEDGER_FIELDS},
        "E_birth_lost": 0.0,
        "eps_hit": None, "x2_hit": None, "side_hit": None, "w_hit": None,
        "kill": None, "emit": None,
    }
    if sp.N == 0:
        return out

    # Kierunek wokół komory zawija się sam w sobie. Pusher czastek potrafi
    # zrobić to od razu przy przesuwaniu. wtedy nie ma tu czego poprawiać.
    if not wrapped:
        sp.x2[:sp.N] = np.mod(sp.ax2, cfg.L_th)

    lo = sp.ax1 < 0.0
    hi = sp.ax1 > cfg.L_r
    hit = lo | hi
    if not np.any(hit):
        return out

    if cfg.wall_mode == "specular":
        # Odbicie sprężyste: czastka wraca do obszaru z odwróconą prędkością w
        # poprzek, nic nie tracąc. Ścianka przestaje być źródłem i ujściem.
        idx = np.nonzero(hit)[0]
        x = sp.x1[idx]
        x = np.where(x < 0.0, -x, 2.0 * cfg.L_r - x)
        
        sp.x1[idx] = np.clip(x, 1e-12, cfg.L_r - 1e-12)
        sp.v1[idx] = -sp.v1[idx]
        
        if sp.tracked:
            sp.n_wall[idx] += 1
        out["w_lo"] = float(np.sum(sp.aw[lo]))
        out["w_hi"] = float(np.sum(sp.aw[hi]))
        
        return out

    idx = np.nonzero(hit)[0]
    eps = 0.5 * sp.mass * (sp.v1[idx]**2 + sp.v2[idx]**2 + sp.v3[idx]**2) / E_CHARGE
    
    w = sp.w[idx].copy()
    x2 = sp.x2[idx].copy()
    x3 = sp.x3[idx].copy()
    
    side = np.where(sp.x1[idx] < 0.0, 0, 1)

    out["w_lo"] = float(np.sum(sp.aw[lo]))
    out["w_hi"] = float(np.sum(sp.aw[hi]))
    out["E_out_eV"] = float(np.sum(w * eps))
    out["q_wall"] = sp.charge * float(np.sum(w))
    out["eps_hit"] = eps
    out["x2_hit"] = x2
    out["side_hit"] = side
    out["w_hit"] = w
    out["kill"] = hit

    if sp.tracked:
        for a in LEDGER_FIELDS:
            out["ledger_lost"][a] = float(np.sum(w * getattr(sp, a)[idx]))
        out["E_birth_lost"] = float(np.sum(w * sp.E_birth[idx]))

    # Emisja elektronów wtórnych. Losujemy, czy dana czastka wybije elektron, a
    # nie dzielimy wagi, dzięki czemu wszystkie czastki modelowe zachowują tę
    # samą wagę i dalsze zderzenia kulombowskie pozostają proste.
    if allow_see and cfg.wall_mode == "see" and sp.charge < 0.0:
        gamma = see_yield(eps, cfg)
        emit = rng.random(idx.size) < gamma
        
        if np.any(emit):
            sel = np.nonzero(emit)[0]
            out["emit"] = {"sel": sel, "side": side[sel], "x2": x2[sel],
                           "x3": x3[sel], "w": w[sel]}
    return out


def emit_secondaries(sp, cfg, payload, rng, t_now, out=None):
    """Dopisuje elektrony wtórne wybite ze ścianki i rozlicza ich energię."""
    if payload is None:
        return 0.0, 0.0
    
    n_new = payload["sel"].size
    
    v_perp, v_a, v_b = _emit_speeds(n_new, cfg.see_T_eV, rng)
    s = payload["side"]
    
    v1 = np.where(s == 0, v_perp, -v_perp)
    x1 = np.where(s == 0, 1e-3 * cfg.h_r, cfg.L_r - 1e-3 * cfg.h_r)
    
    # Elektron wtórny rodzi się tam, gdzie padł ten, który go wybił, także
    # wzdłuż osi komory.
    
    sp.add(x1, payload["x2"], v1, v_a, v_b, payload["w"],
           orig=ORIG_WALL, t_now=t_now, x3=payload["x3"])
    
    w_see = float(np.sum(payload["w"]))
    E_see = float(np.sum(payload["w"] * 0.5 * sp.mass
                         * (v1**2 + v_a**2 + v_b**2) / E_CHARGE))
    if out is not None:
        out["w_see"] = w_see
        out["n_see"] = int(n_new)
        out["E_see_eV"] = E_see
    return w_see, E_see


def apply_boundaries(sp, cfg, rng, t_now, allow_see=True):
    """Jak wyżej, ale od razu usuwa i dosyła czastki."""
    out = wall_events(sp, cfg, rng, t_now, allow_see=allow_see)
    
    if out["kill"] is not None:
        sp.remove_mask(out["kill"], stable=getattr(cfg, "stable_compact", False))
        emit_secondaries(sp, cfg, out["emit"], rng, t_now, out=out)
    return out
