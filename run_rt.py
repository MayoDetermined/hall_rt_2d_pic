"""Uruchamianie pojedynczego przebiegu w poprzek i wokół komory.

Przykłady wywołania:
    python run_rt.py                                 # domyslne
    python run_rt.py --live                          # z podglądem na ywo // mozna sb popatrzec na zmiany 
    python run_rt.py --walls absorb --field mean_only  # bez emisji wtórnej i bez turbulencji
    python run_rt.py --no-ee                         # bez zderzeń między elektronami
    python run_rt.py --Ez 4e4 --t-end 5e-7           # mocniejszy napęd krótszy przebieg
    python run_rt.py --headless klatki/              # zapis klatek zamiast okna
"""

import argparse
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from hall_pic_rt import ConfigRT, SimulationRT, origin
from hall_pic_rt.diagnostics import LiveViewRT


def build_config(args):
    """Składa konfigurację z nastaw podanych w wierszu poleceń."""
    cfg = ConfigRT()

    if args.nr is not None: cfg.N_r = args.nr
    if args.nth is not None: cfg.N_th = args.nth
    if args.ppc is not None: cfg.n_ppc = args.ppc
    if args.dt is not None: cfg.dt = args.dt
    if args.t_end is not None: cfg.t_end = args.t_end
    if args.Ez is not None: cfg.E_z = args.Ez
    if args.Br is not None: cfg.B_r = args.Br
    if args.seed is not None: cfg.seed = args.seed
    if args.diag is not None: cfg.diag_interval = args.diag

    cfg.wall_mode = args.walls
    cfg.field_mode = args.field
    cfg.enable_coulomb_ee = not args.no_ee
    cfg.source_mode = args.source

    if args.axial is not None:
        cfg.axial_length = args.axial

    cfg.enable_mcc = args.mcc
    cfg.wall_bc = args.wall_bc
    cfg.plasma_source = args.plasma_source
    cfg.circuit_mode = args.circuit

    if args.wall_thickness is not None:
        cfg.wall_thickness = args.wall_thickness

    if args.n_neutral is not None:
        cfg.n_neutral = args.n_neutral

    if args.V_ps is not None:
        cfg.V_ps = args.V_ps

    if args.I_set is not None:
        cfg.I_set = args.I_set

    if args.ion_subcycle is not None:
        cfg.ion_subcycle = args.ion_subcycle

    if args.realistic:
        # Gęstość gazu startuje od wartości, przy
        # której jonizacja mniej więcej równoważy straty. bez tego
        # regulator musiałby dojść do niej z daleka i zjadłby na to spory kawałek
        # runu.
        cfg.wall_bc = "floating"
        cfg.plasma_source = "ionization"
        cfg.circuit_mode = "voltage"
        if args.n_neutral is None:
            cfg.n_neutral = 2.0e20
    cfg.live_view = args.live
    if args.headless:
        cfg.headless_save_dir = args.headless
        cfg.live_view = False
    cfg.__post_init__()
    return cfg


def print_header(cfg, n_steps):
    """Wypisuje nastawy i warunki poprawności liczenia przed startem."""
    print("=" * 72)
    print(" PIC 2D3V płaszczyzna w poprzek i wokoł komory")
    print("=" * 72)
    print(f" w poprzek : {cfg.L_r*1e3:.2f} mm, N = {cfg.N_r}, h = {cfg.h_r*1e6:.1f} um"
          f"   [ścianki: {cfg.wall_mode}]")
    print(f" wokół     : {cfg.L_th*1e3:.2f} mm, N = {cfg.N_th}, h = {cfg.h_th*1e6:.1f} um"
          f"   [zawija się]")
    print(f" B w poprzek: {cfg.B_r*1e4:.0f} G      E wzdłuż osi: {cfg.E_z:.3e} V/m")
    print(f" dryf E/B   : {cfg.v_drift:.3e} m/s   promień Larmora: {cfg.rho_e*1e3:.3f} mm")
    print(f" n0 = {cfg.n0:.2e} m^-3, Te0 = {cfg.Te0_eV:.1f} eV, "
          f"dosył = {cfg.source_rate:.3e} m^-3 s^-1")
    print(f" dt = {cfg.dt*1e12:.2f} ps, kroków = {n_steps}, t_end = {cfg.t_end*1e6:.3f} us")
    print(f" czastki   : {cfg.n_cells*cfg.n_ppc} / gatunek (ppc = {cfg.n_ppc})")
    print(f" pole      : {cfg.field_mode}    e-e: {'tak' if cfg.enable_coulomb_ee else 'nie'}"
          f"    gaz: {'tak' if cfg.enable_mcc else 'nie'}")
    print(f" dosył     : {cfg.source_mode}   ucieczka osiowa: "
          + (f"{cfg.axial_length*1e3:.1f} mm" if cfg.axial_length > 0 else "wyłączona"))
    print(" warunki poprawności:")
    for k, v in cfg.checks().items():
        print(f"   {k:<22s} = {v:8.3f}")
    print("=" * 72)


def main():
    """Czyta settingsy, prowadzi przebieg i wypisuje wynik końcowy."""
    p = argparse.ArgumentParser(description="PIC 2D3V w poprzek i wokół komory")
    p.add_argument("--walls", choices=["see", "absorb", "specular"], default="see",
                   help="zachowanie ścianek")
    p.add_argument("--field", choices=["full", "mean_only"], default="full",
                   help="pełne pole albo samo uśrednione, czyli bez turbulencji")
    p.add_argument("--no-ee", action="store_true", help="bez zderzeń elektron-elektron")
    p.add_argument("--source", choices=["fixed", "controlled"], default="controlled",
                   help="dosył o stałej wydajności albo trzymający zadaną gęstość")
    p.add_argument("--axial", type=float, default=1.0e-2,
                   help="długość obszaru wzdłuż osi komory; zero wyłącza ucieczkę osiową")
    p.add_argument("--mcc", action="store_true", help="ze zderzeniami z gazem obojętnym")
    p.add_argument("--wall-bc", choices=["fixed", "floating"], default="fixed",
                   help="ścianka na zadanym potencjale albo pływająca ceramika")
    p.add_argument("--wall-thickness", type=float, default=None,
                   help="grubość warstwy ceramicznej przy ściance pływającej [m]")
    p.add_argument("--plasma-source", choices=["pairs", "ionization"], default="pairs",
                   help="dosył gotowych par albo pełna jonizacja gazu")
    p.add_argument("--n-neutral", type=float, default=None,
                   help="gęstość gazu roboczego [1/m^3]")
    p.add_argument("--circuit", choices=["off", "voltage", "current"], default="off",
                   help="pole wzdłuż osi zadane z góry albo wyliczane z obwodu")
    p.add_argument("--V-ps", type=float, default=None,
                   help="napięcie zasilacza na liczonym odcinku przyspieszania [V]")
    p.add_argument("--I-set", type=float, default=None,
                   help="zadany prąd wyładowania [A]; zero oznacza trzymanie tego, który popłynie")
    p.add_argument("--ion-subcycle", type=int, default=None,
                   help="co ile kroków pchać jony (jeden oznacza co krok)")
    p.add_argument("--realistic", action="store_true",
                   help="wszystkie trzy uściślenia naraz: ścianka pływająca, "
                        "pełna jonizacja i pole z obwodu")
    p.add_argument("--nr", type=int, default=None)
    p.add_argument("--nth", type=int, default=None)
    p.add_argument("--ppc", type=int, default=None)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--t-end", type=float, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--Ez", type=float, default=None)
    p.add_argument("--Br", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--diag", type=int, default=None)
    p.add_argument("--live", action="store_true")
    p.add_argument("--headless", type=str, default="")
    args = p.parse_args()

    cfg = build_config(args)
    n_steps = args.steps if args.steps is not None else cfg.n_steps
    print_header(cfg, n_steps)

    sim = SimulationRT(cfg)
    view = LiveViewRT(cfg)
    t0 = time.time()

    def progress(s, rec):
        """Wypisuje linijkę stanu i odświeża podgląd."""
        wall = time.time() - t0
        print(f" {s.step:8d}  t={s.t*1e9:8.1f} ns  N_e={s.electrons.N:7d}"
              f"  dn/n={rec['dn_rel']:.4f}  E~θ={rec['E_th_rms']:9.1f} V/m"
              f"  Te={rec['Te_bulk']:6.2f} eV  krotność={rec['excess_ratio']:7.3f}"
              f"  [{s.step/max(wall,1e-9):5.1f} kr/s]")
        if view.enabled and s.step % cfg.plot_interval == 0:
            view.update(s)

    sim.run(n_steps, progress=progress)

    print("=" * 72)
    print(f" Zakończono: {sim.step} kroków, t = {sim.t*1e9:.1f} ns, "
          f"czas rzeczywisty = {time.time()-t0:.0f} s")
    print("=" * 72)
    print(sim.stats_text())
    if view.enabled:
        view.update(sim)
        view.finalize()


if __name__ == "__main__":
    main()
