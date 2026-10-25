"""Odczytywanie metryczki pochodzenia i miara nadwyżki szybkich elektronów.

Sama obecność szybkich elektronów niczego nie rozstrzyga, bo pewna ich liczba
wynika już z samego rozkładu cieplnego. Interesuje nas nadwyżka, czyli o ile
więcej jest ich naprawdę, niż wynikałoby z tła. Dlatego do części rozkładu
poniżej zadanej energii dopasowujemy rozkład cieplny, przedłużamy go w górę i
porównujemy z tym, co jest naprawdę.

Drugie pytanie brzmi, skąd te elektrony wzięły energię. Odpowiada na nie
metryczka, którą każdy elektron nosi ze sobą: osobny rachunek pracy wykonanej
nad nim przez każdą składową pola, licznik zetknięć ze ścianką i znacznik
pochodzenia. Zsumowana po samych szybkich elektronach daje wprost udział obu
mechanizmów.
"""

import numpy as np
from scipy.special import gammainc, gamma as gamma_fn
from scipy.optimize import brentq

from .species import (LEDGER_FIELDS, LEDGER_LABELS, ORIGIN_NAMES, ORIG_WALL)


def eedf(eps, weights, bins):
    """Zwraca rozkład energii elektronów: liczności w zadanych przedziałach."""
    h, edges = np.histogram(eps, bins=bins, weights=weights)
    centers = 0.5 * (edges[1:] + edges[:-1])
    width = np.diff(edges)
    return centers, h / np.maximum(width, 1e-300)


def fit_bulk_temperature(eps, weights, eps_cut):
    """Dopasowuje temperaturę tła do części rozkładu poniżej zadanej energii.

    Korzystamy z tego, że dla rozkładu cieplnego średnia energia liczona tylko
    do pewnej granicy jest jednoznaczną funkcją temperatury. Wystarczy więc
    zmierzyć tę średnią i rozwiązać równanie względem temperatury.
    """
    m = eps < eps_cut
    if not np.any(m):
        return np.nan
    
    w = weights[m]
    mean_eps = float(np.sum(w * eps[m]) / np.sum(w))

    def truncated_mean(T):
        x = eps_cut / T
        
        lo = gammainc(1.5, x)
        hi = gammainc(2.5, x) # hi 
        
        if lo <= 0.0:
            return np.inf
        return 1.5 * T * hi / lo

    try:
        return brentq(lambda T: truncated_mean(T) - mean_eps,
                      1e-3 * eps_cut, 50.0 * eps_cut, xtol=1e-6, rtol=1e-8)
    except Exception:
        return np.nan


def excess_metrics(eps, weights, cfg):
    """Mierzy nadwyzke szybkich elektronów względem tła cieplnego.

    Zwraca zmierzony udział szybkich elektronów, udział wynikający z samego tła,
    ich różnicę oraz stosunek ile jest tych runaways niz powinno
    """
    out = {"Te_bulk": np.nan, "f_fast": 0.0, "f_fast_maxwell": np.nan,
           "excess": np.nan, "excess_ratio": np.nan,
           "f_energy_fast": 0.0, "mean_eps": 0.0, "n_e_tot": 0.0}
    
    if eps.size == 0:
        return out
    
    w_tot = float(np.sum(weights))
    out["n_e_tot"] = w_tot
    out["mean_eps"] = float(np.sum(weights * eps) / w_tot)

    fast = eps > cfg.E_fast_eV
    w_fast = float(np.sum(weights[fast]))
    
    out["f_fast"] = w_fast / w_tot
    out["f_energy_fast"] = float(np.sum(weights[fast] * eps[fast])) / float(
        np.sum(weights * eps))

    T = fit_bulk_temperature(eps, weights, cfg.eedf_bulk_eV)
    out["Te_bulk"] = T
    
    if not np.isfinite(T) or T <= 0.0:
        return out

    # Rozkład cieplny znormowany tak, żeby zgadzał się z tłem poniżej granicy
    # dopasowania, a potem przedłużamy go powyżej progu szybkich elektronów.
    
    p_bulk = gammainc(1.5, cfg.eedf_bulk_eV / T)
    if p_bulk <= 0.0:
        return out
    
    w_tot_maxwell = float(np.sum(weights[eps < cfg.eedf_bulk_eV])) / p_bulk

    def one(eps_f):
        """Zwraca zmierzony udział, udział z tła i ich stosunek przy jednym progu."""
        wm = float(np.sum(weights[eps > eps_f]))
        we = w_tot_maxwell * (1.0 - gammainc(1.5, eps_f / T))

        return (wm / w_tot, we / w_tot,
                (wm / we) if we > 0 else np.inf)

    f_m, f_e, ratio = one(cfg.E_fast_eV)
    out["f_fast_maxwell"] = f_e
    out["excess"] = f_m - f_e
    out["excess_ratio"] = ratio

    # To samo przy kilku progach stałych i kilku wyrażonych krotnością
    # temperatury tła, żeby sprawdzić, czy wniosek nie zależy od wyboru progu.
    for e_f in cfg.E_fast_list:
        m, e, r = one(e_f)
        out[f"f_fast@{e_f:g}"] = m
        out[f"ratio@{e_f:g}"] = r
        out[f"excess@{e_f:g}"] = m - e
    
    for k in cfg.fast_over_Te:
        m, e, r = one(k * T)
        out[f"ratio@{k:g}Te"] = r
        out[f"excess@{k:g}Te"] = m - e
    
    return out


def attribution(electrons, mask=None):
    """Rozkłada energię wybranej grupy elektronów na mechanizmy, które ją dostarczyły.

    Dla każdej składowej pola podaje łączną pracę wykonaną nad tą grupą oraz jej
    udział w sumie. Osobno podaje udział elektronów pochodzących ze ścianek i
    tych, które zetknęły się ze ścianką choć raz.
    """
    out = {}
    if electrons.N == 0 or not electrons.tracked:
        return out
    
    m = np.ones(electrons.N, dtype=bool) if mask is None else mask
    w = electrons.aw[m]
    
    w_tot = float(np.sum(w))
    
    out["w_tot"] = w_tot
    
    if w_tot <= 0.0:
        return out

    led = electrons.ledger()
    total_pos = 0.0
    
    for a in LEDGER_FIELDS:
        val = float(np.sum(w * led[a][m]))
        out[a] = val
        out[a + "_per_particle"] = val / w_tot
        if val > 0.0:
            total_pos += val
    
    out["W_total"] = sum(out[a] for a in LEDGER_FIELDS)
    
    for a in LEDGER_FIELDS:
        out["frac_" + a] = (out[a] / total_pos) if total_pos > 0 else 0.0

    # Udział mechanizmu ścianek liczony po pochodzeniu, a nie po energii.
    orig = electrons.orig[:electrons.N][m]
    for code, name in ORIGIN_NAMES.items():
        out["orig_" + name] = float(np.sum(w[orig == code])) / w_tot
    
    nw = electrons.n_wall[:electrons.N][m]
    
    out["frac_touched_wall"] = float(np.sum(w[nw > 0])) / w_tot
    out["frac_wall_born"] = float(np.sum(w[orig == ORIG_WALL])) / w_tot

    out["mean_t_birth"] = float(np.sum(w * electrons.t_birth[:electrons.N][m]) / w_tot)
    out["mean_n_wall"] = float(np.sum(w * nw) / w_tot)
    return out


def dominant_channel(electrons, mask=None):
    """Wskazuje dla każdego elektronu, która składowa pola dała mu najwięcej energii.

    Zwraca udział wagi elektronów przypisanych do każdego mechanizmu. Elektron
    trafia tam, gdzie jego własny rachunek energii ma największą pozycję
    dodatnią. jeśli żadna nie jest dodatnia, nie trafia nigdzie.
    """
    if electrons.N == 0 or not electrons.tracked:
        return {}
    
    m = np.ones(electrons.N, dtype=bool) if mask is None else mask
    w = electrons.aw[m]
    
    w_tot = float(np.sum(w))
    
    if w_tot <= 0.0:
        return {}

    led = electrons.ledger()
    stack = np.vstack([led[a][m] for a in LEDGER_FIELDS])
    
    best = np.argmax(stack, axis=0)
    best_val = stack[best, np.arange(best.size)]
    
    none = best_val <= 0.0

    out = {}
    
    for k, a in enumerate(LEDGER_FIELDS):
        out["dom_" + a] = float(np.sum(w[(best == k) & ~none])) / w_tot
    out["dom_none"] = float(np.sum(w[none])) / w_tot
    return out


def summarize(electrons, cfg):
    """Składa pełne podsumowanie: nadwyżka, przypisanie energii i pochodzenie.

    Robi to osobno dla wszystkich elektronów i dla samych szybkich,
    """
    if electrons.N == 0:
        return {}
    
    eps = electrons.kinetic_energy_eV()
    w = electrons.aw
    res = excess_metrics(eps, w, cfg)

    fast = eps > cfg.E_fast_eV
    
    res["attr_all"] = attribution(electrons, cfg)
    res["attr_fast"] = attribution(electrons, cfg, mask=fast)
    res["dom_all"] = dominant_channel(electrons, cfg)
    res["dom_fast"] = dominant_channel(electrons, cfg, mask=fast)

    # To samo przy progu skalowanym temperaturą tła. Przy stałym progu, gdy
    # temperatura różni się między przebiegami kilkukrotnie, ten sam próg raz
    # wycina sam ogon, a raz niemal cały rozkład, więc porównywanie przypisania
    # energii między przebiegami traci sens. Próg będący krotnością temperatury
    # wycina za to w każdym przebiegu tę samą część rozkładu.
    
    T = res.get("Te_bulk", np.nan)
    
    if np.isfinite(T) and T > 0:
        for k in cfg.fast_over_Te:
            m = eps > k * T
            if np.any(m):
                res[f"attr_fast@{k:g}Te"] = attribution(electrons, cfg, mask=m)

    if electrons.tracked and np.any(fast):
        res["fast_wall_born"] = res["attr_fast"].get("frac_wall_born", 0.0)
        res["fast_touched_wall"] = res["attr_fast"].get("frac_touched_wall", 0.0)
        
        resid = electrons.ledger_residual()
        
        typ = max(float(np.mean(np.abs(eps))), 1e-30)
        res["ledger_residual_rel"] = float(np.mean(np.abs(resid))) / typ
    
    return res


def format_report(res, cfg):
    """Składa czytelny opis wyniku, gotowy do wypisania na ekran."""
    if not res:
        return "brak elektronów"
    lines = []
    lines.append(f"T_e tła            : {res['Te_bulk']:.2f} eV")
    lines.append(f"<E_e>              : {res['mean_eps']:.2f} eV")
    lines.append(f"udział E > {cfg.E_fast_eV:.0f} eV : {res['f_fast']*100:.3f} %")
    lines.append(f"  z samego tła     : {res['f_fast_maxwell']*100:.3f} %")
    lines.append(f"  nadwyżka         : {res['excess']*100:.3f} pkt proc.")
    lines.append(f"  ile razy więcej  : {res['excess_ratio']:.2f}")
    lines.append(f"udział energii ogona: {res['f_energy_fast']*100:.2f} %")
    af = res.get("attr_fast", {})
    if af:
        lines.append("skąd energia szybkich elektronów:")
        for a in LEDGER_FIELDS:
            if a in af:
                lines.append(f"  {LEDGER_LABELS[a]:<30s} {af['frac_'+a]*100:6.2f} %"
                             f"  ({af[a+'_per_particle']:8.2f} eV/czastkę)")
        lines.append(f"  wybite ze ścianki             {af.get('frac_wall_born',0)*100:6.2f} %")
        lines.append(f"  dotknęły ścianki              {af.get('frac_touched_wall',0)*100:6.2f} %")
    if "ledger_residual_rel" in res:
        lines.append(f"reszta metryczki    : {res['ledger_residual_rel']:.2e}")
    return "\n".join(lines)
