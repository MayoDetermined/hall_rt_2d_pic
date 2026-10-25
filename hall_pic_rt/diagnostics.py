"""Podgląd przebiegu: mapy, rozkład energii i przebiegi w czasie.

Sześć okienek dobrano tak, żeby dało się jednym spojrzeniem ocenić, czy w
przebiegu dzieje się to, co miało się dziać. Na mapie gęstości i na mapie
zaburzeń pola wokół komory widać, czy rozwinęła się niestabilność i jaką ma
długość fali. Na profilu w poprzek komory widać warstwy przy ściankach. Na
rozkładzie energii widać ogon razem z tłem cieplnym, do którego go porównujemy,
a na przebiegach w czasie to, czy przebieg zdążył się ustalić.
"""

import os
import numpy as np

from . import origin
from .pusher import deposit_number_density


class LiveViewRT:
    """Rysuje podgląd przebiegu albo zapisuje kolejne klatki do plików."""

    def __init__(self, cfg):
        """Zakłada okno z wykresami albo katalog na klatki."""
        self.cfg = cfg
        self.enabled = cfg.live_view or bool(cfg.headless_save_dir)
        if not self.enabled:
            return
        import matplotlib
        if cfg.headless_save_dir:
            matplotlib.use("Agg")
            os.makedirs(cfg.headless_save_dir, exist_ok=True)
        import matplotlib.pyplot as plt
        self.plt = plt
        if cfg.live_view and not cfg.headless_save_dir:
            plt.ion()
        self.fig, self.axes = plt.subplots(2, 3, figsize=(17, 9))
        self.fig.suptitle(
            f"PIC 2D3V w poprzek i wokół komory — ścianki: {cfg.wall_mode}, "
            f"pole: {cfg.field_mode}, e-e: {'tak' if cfg.enable_coulomb_ee else 'nie'}")
        self._cb = [None, None]

    def _extent(self):
        """Zakres obu osi mapy w milimetrach."""
        cfg = self.cfg
        return [0.0, cfg.L_th * 1e3, 0.0, cfg.L_r * 1e3]

    def update(self, sim):
        """Przerysowuje wszystkie okienka na podstawie bieżącego stanu."""
        if not self.enabled:
            return
        cfg, ax = self.cfg, self.axes
        el = sim.electrons

        # Gęstość elektronów na płaszczyźnie.
        a = ax[0, 0]; a.clear()
        ne = deposit_number_density(el, cfg)
        im = a.imshow(ne, origin="lower", aspect="auto", extent=self._extent(),
                      cmap="viridis")
        a.set_title(f"n_e   t = {sim.t*1e9:.1f} ns")
        a.set_xlabel("wokół komory [mm]"); a.set_ylabel("w poprzek komory [mm]")
        if self._cb[0] is None:
            self._cb[0] = self.fig.colorbar(im, ax=a, fraction=0.046)
        else:
            self._cb[0].update_normal(im)

        # Zaburzenie pola wokół komory, czyli sama turbulencja.
        a = ax[0, 1]; a.clear()
        lim = max(np.max(np.abs(sim.E_th_fluc)), 1e-9)
        im2 = a.imshow(sim.E_th_fluc, origin="lower", aspect="auto",
                       extent=self._extent(), cmap="coolwarm", vmin=-lim, vmax=lim)
        a.set_title("zaburzenie pola wokół komory [V/m]")
        a.set_xlabel("wokół komory [mm]"); a.set_ylabel("w poprzek komory [mm]")
        if self._cb[1] is None:
            self._cb[1] = self.fig.colorbar(im2, ax=a, fraction=0.046)
        else:
            self._cb[1].update_normal(im2)

        # Profile w poprzek komory.
        a = ax[0, 2]; a.clear(); a.grid(alpha=0.3)
        pr = sim.radial_profiles()
        r_mm = pr["r"] * 1e3
        a.plot(r_mm, pr["n_e"] / 1e16, "b-", label="n_e [10¹⁶ m⁻³]")
        a.plot(r_mm, pr["Te"], "r-", label="T_e [eV]")
        a2 = a.twinx(); a2.plot(r_mm, pr["phi"], "k--", lw=1, label="φ")
        a2.set_ylabel("φ [V]")
        a.set_title("profile w poprzek komory"); a.set_xlabel("r [mm]")
        a.legend(fontsize=8, loc="upper center")

        # Rozkład energii z dopasowanym tłem cieplnym.
        a = ax[1, 0]; a.clear(); a.grid(alpha=0.3)
        eps = el.kinetic_energy_eV()
        if eps.size:
            top = max(4.0 * cfg.E_fast_eV, float(np.percentile(eps, 99.99)))
            bins = np.linspace(0.0, top, 120)
            c, f = origin.eedf(eps, el.aw, bins)
            a.semilogy(c, np.maximum(f, 1e-30), "b-", label="zmierzony")
            res = origin.excess_metrics(eps, el.aw, cfg)
            T = res["Te_bulk"]
            if np.isfinite(T):
                ref = np.sqrt(c) * np.exp(-c / T)
                m = c < cfg.eedf_bulk_eV
                scale = np.sum(f[m]) / max(np.sum(ref[m]), 1e-300)
                a.semilogy(c, ref * scale, "r--", label=f"tło cieplne {T:.1f} eV")
            a.axvline(cfg.E_fast_eV, color="k", ls=":", label=f"próg {cfg.E_fast_eV:.0f} eV")
            a.set_ylim(max(f.max(), 1e-30) * 1e-7, f.max() * 3)
            a.legend(fontsize=8)
        a.set_title("rozkład energii elektronów"); a.set_xlabel("energia [eV]")

        # Przebiegi w czasie.
        a = ax[1, 1]; a.clear(); a.grid(alpha=0.3)
        if sim.history:
            t = np.array([h["t"] for h in sim.history]) * 1e9
            a.semilogy(t, [max(h["E_th_rms"], 1e-9) for h in sim.history],
                       "g-", label="zaburzenie pola [V/m]")
            a3 = a.twinx()
            a3.plot(t, [h["excess_ratio"] for h in sim.history], "m-",
                    label="ile razy więcej szybkich")
            a3.axhline(1.0, color="gray", lw=0.5)
            a3.set_ylabel("nadwyżka (krotność)")
            a.legend(fontsize=8, loc="upper left")
        a.set_title("przebieg w czasie"); a.set_xlabel("t [ns]")

        # Podsumowanie liczbowe.
        a = ax[1, 2]; a.clear(); a.axis("off")
        a.text(0.0, 1.0, sim.stats_text(), va="top", ha="left",
               family="monospace", fontsize=6.8, transform=a.transAxes)

        self.fig.tight_layout()
        if self.cfg.headless_save_dir:
            self.fig.savefig(os.path.join(self.cfg.headless_save_dir,
                                          f"frame_{sim.step:07d}.png"), dpi=80)
        else:
            self.fig.canvas.draw(); self.fig.canvas.flush_events()
            self.plt.pause(0.001)

    def finalize(self):
        """Zostawia otwarte okno na końcu przebiegu, o ile pracujemy z ekranem."""
        if not self.enabled or self.cfg.headless_save_dir:
            return
        self.plt.ioff(); self.plt.show()
