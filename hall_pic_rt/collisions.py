"""Zderzenia czastek z gazem, liczone metodą zderzeń zerowych.

 Najpierw ustalamy największą możliwą częstość zderzeń w całym zakresie
energii. Z niej wynika, jaki ułamek czastek w ogóle bierze udział w losowaniu,
więc losujemy tylko tę garstkę, co jest tanie. Dla każdej wylosowanej czastki
liczymy dopiero prawdziwe szanse na poszczególne rodzaje zderzeń. Jeśli żadne
nie wypadnie, mówimy, że zderzenie było zerowe, i nic się nie dzieje.

Po każdym zderzeniu obracamy prędkość czastki w losowym kierunku, a jej długość
dobieramy zgodnie z tym, ile energii dane zderzenie zabiera.
"""

import numpy as np

from .constants import E_CHARGE, M_ELECTRON, K_BOLTZMANN
from . import cross_sections as xs


def isotropic_unit_vectors(n, rng):
    """Zwraca zadaną liczbę losowych kierunków rozłożonych równo we wszystkie strony."""
    cos_theta = 2.0 * rng.random(n) - 1.0
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
    phi = 2.0 * np.pi * rng.random(n)
    return (sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta)


class NullCollisionMCC:
    """Obsługuje zderzenia elektronów i jonów z gazem metodą zderzeń zerowych."""

    def __init__(self, cfg, rng):
        """Zapamiętuje ustawienia i generator losowy oraz przygotowuje częstości."""
        self.cfg = cfg
        self.rng = rng
        self._prepare_electron()
        self._prepare_ion()

    # Ustalenie największej możliwej częstości zderzeń i wynikającej z niej
    # szansy, że dana czastka trafi do losowania.
    def _prepare_electron(self):
        """Wyznacza dla elektronów największą częstość zderzeń i związaną z nią szansę losowania."""
        
        eps = np.linspace(0.01, self.cfg.eps_max_grid_eV, self.cfg.n_energy_grid)
        v = xs.speed_from_energy_e(eps)

        n_n_max = self.cfg.n_n_anode
        rate = n_n_max * xs.sigma_total_e(eps) * v

        self.nu_max_e = float(np.max(rate))

        self.P_e = 1.0 - np.exp(-self.nu_max_e * self.cfg.dt)

    def _prepare_ion(self):
        """Wyznacza dla jonów największą częstość zderzeń i związaną z nią szansę losowania."""
        eps = np.linspace(0.01, self.cfg.eps_max_grid_eV, self.cfg.n_energy_grid)
        v = np.sqrt(2.0 * eps * E_CHARGE / self.cfg.m_ion)

        n_n_max = self.cfg.n_n_anode
        rate = n_n_max * xs.sigma_total_ion(eps) * v

        self.nu_max_i = float(np.max(rate))

        self.P_i = 1.0 - np.exp(-self.nu_max_i * self.cfg.dt)

    # Zderzenia elektronów.
    def collide_electrons(self, electrons, ions, w_ref):
        """Ogarnia zderzenia elektronów i zwraca, ile powstało przy tym nowych jonów.

        rozpatujemy, ze wylosowane elektrony mogą odbić się sprężyście, wzbudzić atom albo go
        zjonizować i stworzyc nową parę
        """
        cfg = self.cfg
        N = electrons.N
        if N == 0 or self.P_e <= 0.0:
            return 0
        rng = self.rng

        # Wybieramy losowo tę część elektronów, która bierze udział w losowaniu.
        n_cand = rng.binomial(N, self.P_e)

        if n_cand == 0:
            return 0
        idx = rng.choice(N, size=n_cand, replace=False)

        vx = electrons.vx[idx]; vy = electrons.vy[idx]; vz = electrons.vz[idx]
        speed = np.sqrt(vx*vx + vy*vy + vz*vz)
        eps = 0.5 * M_ELECTRON * speed**2 / E_CHARGE
        n_n = cfg.neutral_density(electrons.x[idx])

        inv_num = 1.0 / self.nu_max_e
        nu_el = n_n * xs.sigma_elastic_e(eps) * speed * inv_num
        nu_ex = n_n * xs.sigma_excitation_e(eps) * speed * inv_num
        nu_iz = n_n * xs.sigma_ionization_e(eps) * speed * inv_num

        r = rng.random(n_cand)
        c1 = nu_el
        c2 = nu_el + nu_ex
        c3 = nu_el + nu_ex + nu_iz

        is_el = r < c1
        is_ex = (r >= c1) & (r < c2)
        is_iz = (r >= c2) & (r < c3)
        # Kto wylosował więcej niż suma szans, trafił na zderzenie zerowe i nic go nie zmienia.

        n_ion = 0
        # Zderzenie sprężyste: energia niemal się nie zmienia, bo atom jest
        # o wiele cięższy, ale kierunek prędkości ustawiamy losowo.

        if np.any(is_el):
            sel = np.nonzero(is_el)[0]
            dvx, dvy, dvz = isotropic_unit_vectors(sel.size, rng)

            # Elektron oddaje atomowi tylko drobny ułamek energii.
            dE_frac = 2.0 * M_ELECTRON / cfg.m_ion
            new_eps = np.maximum(eps[sel] * (1.0 - dE_frac), 1e-3)
            new_speed = xs.speed_from_energy_e(new_eps)

            self._set_iso(electrons, idx[sel], new_speed, dvx, dvy, dvz)

        # Wzbudzenie: elektron oddaje energię progu wzbudzenia i leci dalej w losowym kierunku.
        if np.any(is_ex):
            sel = np.nonzero(is_ex)[0]

            new_eps = np.maximum(eps[sel] - xs.E_EXC_XE, 1e-3)
            new_speed = xs.speed_from_energy_e(new_eps)

            dvx, dvy, dvz = isotropic_unit_vectors(sel.size, rng)

            self._set_iso(electrons, idx[sel], new_speed, dvx, dvy, dvz)

        # Jonizacja: z jednego elektronu robią się dwa, a w kanale przybywa jon.
        if np.any(is_iz):
            sel = np.nonzero(is_iz)[0]
            gi = idx[sel]
            avail = np.maximum(eps[sel] - xs.E_ION_XE, 0.0)

            # Energię, która zostaje po pokryciu progu, dzielimy losowo między
            # elektron pierwotny i wybity.

            split = rng.random(sel.size)
            e_prim = avail * split
            e_sec = avail * (1.0 - split)

            # Elektron pierwotny leci dalej w nowym, losowym kierunku.
            sp1 = xs.speed_from_energy_e(np.maximum(e_prim, 1e-3))
            d1 = isotropic_unit_vectors(sel.size, rng)
            self._set_iso(electrons, gi, sp1, *d1)

            # Elektron wybity to nowa czastka o tej samej wadze co rodzic.
            wpar = electrons.w[gi]
            sp2 = xs.speed_from_energy_e(np.maximum(e_sec, 1e-3))
            d2 = isotropic_unit_vectors(sel.size, rng)

            xpos = electrons.x[gi]
            electrons.add(xpos, sp2*d2[0], sp2*d2[1], sp2*d2[2], wpar)

            # Powstały jon startuje z miejsca jonizacji z prędkością termiczną gazu.
            vth_i = np.sqrt(K_BOLTZMANN * cfg.T_neutral / cfg.m_ion)

            ivx = rng.normal(0.0, vth_i, sel.size)
            ivy = rng.normal(0.0, vth_i, sel.size)
            ivz = rng.normal(0.0, vth_i, sel.size)
            
            ions.add(xpos.copy(), ivx, ivy, ivz, wpar.copy())
            n_ion = sel.size

        return n_ion

    def _set_iso(self, sp, gidx, new_speed, ux, uy, uz):
        """Ustawia prędkość wskazanych czastek na nową długość i losowy kierunek."""
        sp.vx[gidx] = new_speed * ux
        sp.vy[gidx] = new_speed * uy
        sp.vz[gidx] = new_speed * uz

    # Zderzenia jonów z zamrożonym gazem.
    def collide_ions(self, ions, w_ref):
        """Rozgrywa zderzenia jonów z atomami: wymianę ładunku i odbicia sprężyste."""
        cfg = self.cfg
        N = ions.N
        if N == 0 or self.P_i <= 0.0:
            return
        
        rng = self.rng
        n_cand = rng.binomial(N, self.P_i)

        if n_cand == 0:
            return
        
        idx = rng.choice(N, size=n_cand, replace=False)
        vx = ions.vx[idx]; vy = ions.vy[idx]; vz = ions.vz[idx]

        # Losujemy prędkość atomu, z którym jon się spotyka.
        vth_n = np.sqrt(K_BOLTZMANN * cfg.T_neutral / cfg.m_ion)
        nvx = rng.normal(0.0, vth_n, n_cand)
        nvy = rng.normal(0.0, vth_n, n_cand)
        nvz = rng.normal(0.0, vth_n, n_cand)

        # Liczy się prędkość jonu względem tego atomu.
        gx = vx - nvx; gy = vy - nvy; gz = vz - nvz
        g = np.sqrt(gx*gx + gy*gy + gz*gz)
        eps_rel = 0.5 * cfg.m_ion * g**2 / E_CHARGE

        n_n = cfg.neutral_density(ions.x[idx])
        inv_num = 1.0 / self.nu_max_i

        nu_cex = n_n * xs.sigma_cex_ion(eps_rel) * g * inv_num
        nu_el = n_n * xs.sigma_elastic_ion(eps_rel) * g * inv_num

        r = rng.random(n_cand)
        is_cex = r < nu_cex
        is_el = (r >= nu_cex) & (r < nu_cex + nu_el)

        # Przy wymianie ładunku jon przejmuje prędkość atomu, czyli praktycznie stygnie.
        if np.any(is_cex):
            sel = np.nonzero(is_cex)[0]
            ions.vx[idx[sel]] = nvx[sel]
            ions.vy[idx[sel]] = nvy[sel]
            ions.vz[idx[sel]] = nvz[sel]

        # Przy odbiciu sprężystym jon i atom mają równe masy, więc obracamy
        # ich ruch względny w losowym kierunku wokół wspólnego środka masy.
        if np.any(is_el):
            sel = np.nonzero(is_el)[0]
            ux, uy, uz = isotropic_unit_vectors(sel.size, rng)
            
            gsel = g[sel]
            # Prędkość środka masy to średnia prędkości jonu i atomu.
            cmx = 0.5*(vx[sel] + nvx[sel]); cmy = 0.5*(vy[sel] + nvy[sel]); cmz = 0.5*(vz[sel] + nvz[sel])

            ions.vx[idx[sel]] = cmx + 0.5*gsel*ux
            ions.vy[idx[sel]] = cmy + 0.5*gsel*uy
            ions.vz[idx[sel]] = cmz + 0.5*gsel*uz
