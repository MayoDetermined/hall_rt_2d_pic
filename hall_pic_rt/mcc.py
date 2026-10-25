"""Zderzenia elektronów z zamrożonym gazem obojętnym.

aspekt fizyczny jest talki sam co w wersji jednowymiarowej i korzysta z tych samych
prawdopodobieństw zderzeń dla ksenonu. Zmienia się tylko to, że gęstość gazu
jest tu jednorodna, bo płaszczyzna obliczeń leży w poprzek i wokół komory, a nie
wzdłuż jej osi.

Zderzenia z gazem są dla szybkich elektronów przede wszystkim stratą: wzbudzenie
i jonizacja odbierają im energię paczkami po kilkanaście elektronowoltów, czyli
dokładnie w skali samego ogona. Dlatego są tu osobno rozliczane w metryczce, aby
dało się oddzielić to, ile energii mechanizmy dostarczyły, od tego, ile z niej
gaz zdążył zabrać.
"""

import numpy as np

from hall_pic_rt.constants import E_CHARGE, M_ELECTRON, K_BOLTZMANN
from hall_pic_rt import cross_sections as xs
from hall_pic_rt.collisions import isotropic_unit_vectors
from .species import ORIG_IONIZ


class NeutralMCC:
    """Rozgrywa zderzenia elektronów z gazem metodą zderzeń zerowych."""

    def __init__(self, cfg, rng):
        """Wyznacza największą częstość zderzeń i wynikającą z niej szansę losowania."""
        self.cfg = cfg
        self.rng = rng

        eps = np.linspace(0.01, cfg.eps_max_grid_eV, cfg.n_energy_grid)
        v_e = xs.speed_from_energy_e(eps)

        # Największy iloczyn skłonności do zderzenia i prędkości, jeszcze bez
        # gęstości gazu. Dzięki temu zmiana gęstości gazu w trakcie przebiegu
        # kosztuje jedno mnożenie, a nie ponowne przeglądanie całej tablicy.
        self._sv_max = float(np.max(xs.sigma_total_e(eps) * v_e))
        self.n_neutral = float(cfg.n_neutral)

        self._update_rate()

    def _update_rate(self):
        """Przelicza szansę wylosowania czastki do zderzenia dla bieżącej gęstości gazu."""
        self.nu_max = self.n_neutral * self._sv_max
        self.P = 1.0 - np.exp(-self.nu_max * self.cfg.dt)

    def set_density(self, n_neutral):
        """Ustawia bieżącą gęstość gazu roboczego."""
        if n_neutral != self.n_neutral:
            self.n_neutral = float(n_neutral)
            self._update_rate()

    def collide(self, electrons, ions, t_now):
        """Rozgrywa zderzenia i zwraca zmianę energii elektronów oraz liczbę jonizacji."""
        cfg, rng = self.cfg, self.rng
        N = electrons.N

        if N == 0 or self.P <= 0.0:
            return 0.0, 0
        
        n_cand = rng.binomial(N, self.P)
        
        if n_cand == 0:
            return 0.0, 0
        
        idx = rng.choice(N, size=n_cand, replace=False)

        v1 = electrons.v1[idx].copy()
        v2 = electrons.v2[idx].copy()
        
        v3 = electrons.v3[idx].copy()
        speed = np.sqrt(v1*v1 + v2*v2 + v3*v3)
        eps = 0.5 * M_ELECTRON * speed**2 / E_CHARGE
        w = electrons.w[idx].copy()

        inv = 1.0 / self.nu_max
        n_n = self.n_neutral
        
        p_el = n_n * xs.sigma_elastic_e(eps) * speed * inv
        p_ex = n_n * xs.sigma_excitation_e(eps) * speed * inv
        p_iz = n_n * xs.sigma_ionization_e(eps) * speed * inv

        r = rng.random(n_cand)
        
        is_el = r < p_el
        is_ex = (r >= p_el) & (r < p_el + p_ex)
        is_iz = (r >= p_el + p_ex) & (r < p_el + p_ex + p_iz)

        new_eps = eps.copy()
        changed = np.zeros(n_cand, dtype=bool)

        if np.any(is_el):
            s = np.nonzero(is_el)[0]
            dE = 2.0 * M_ELECTRON / cfg.m_ion
            new_eps[s] = np.maximum(eps[s] * (1.0 - dE), 1e-3)
            changed[s] = True

        if np.any(is_ex):
            s = np.nonzero(is_ex)[0]
            new_eps[s] = np.maximum(eps[s] - xs.E_EXC_XE, 1e-3)
            changed[s] = True

        n_ion = 0
        sec_data = None
        if np.any(is_iz):
            s = np.nonzero(is_iz)[0]
            avail = np.maximum(eps[s] - xs.E_ION_XE, 0.0)
            frac = rng.random(s.size)
            new_eps[s] = np.maximum(avail * frac, 1e-3)
            changed[s] = True
            sec_data = (s, np.maximum(avail * (1.0 - frac), 1e-3))
            n_ion = s.size

        if not np.any(changed):
            return 0.0, 0

        ch = np.nonzero(changed)[0]
        gidx = idx[ch]
        u1, u2, u3 = isotropic_unit_vectors(ch.size, rng)
        sp_new = xs.speed_from_energy_e(new_eps[ch])
        electrons.v1[gidx] = sp_new * u1
        electrons.v2[gidx] = sp_new * u2
        electrons.v3[gidx] = sp_new * u3

        dE_eV = new_eps[ch] - eps[ch]
        if electrons.tracked:
            electrons.W_coll[gidx] += dE_eV
        dE_total = float(np.sum(w[ch] * dE_eV))

        # Nowa para z jonizacji. Elektron wtórny dostaje resztę energii, a jon
        # rodzi się zimny, z temperaturą gazu.
        if sec_data is not None:
            s, e_sec = sec_data
            gi = idx[s]
            
            xa = electrons.x1[gi].copy()
            xb = electrons.x2[gi].copy()
            xc = electrons.x3[gi].copy()
            ws = electrons.w[gi].copy()
            
            us = isotropic_unit_vectors(s.size, rng)
            
            v_sec = xs.speed_from_energy_e(e_sec)
            electrons.add(xa, xb, v_sec*us[0], v_sec*us[1], v_sec*us[2], ws,
                          orig=ORIG_IONIZ, t_now=t_now, x3=xc)
            
            vth_i = np.sqrt(K_BOLTZMANN * cfg.T_neutral / cfg.m_ion)
            vi1 = rng.normal(0, vth_i, s.size)
            vi2 = rng.normal(0, vth_i, s.size)
            
            vi3 = rng.normal(0, vth_i, s.size)
            ions.add(xa.copy(), xb.copy(), vi1, vi2, vi3, ws.copy(), x3=xc.copy())
            # Do księgi trafia energia obu nowych czastek. Jon rodzi się zimny,
            # ale jego energia cieplna też skądś pochodzi, więc pominięcie jej
            # zostawiłoby w księdze dziurę.
            e_ion = 0.5 * cfg.m_ion * (vi1**2 + vi2**2 + vi3**2) / E_CHARGE
            dE_total += float(np.sum(ws * (e_sec + e_ion)))

        return dE_total, n_ion
