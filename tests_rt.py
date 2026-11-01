"""Sprawdziany poprawności wersji liczącej w poprzek i wokół komory.

Uruchomienie:
    python tests_rt.py
"""

import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from hall_pic_rt.constants import E_CHARGE, M_ELECTRON, EPS0
from hall_pic_rt import ConfigRT, SimulationRT
from hall_pic_rt.poisson import PoissonRT
from hall_pic_rt.species import SpeciesRT
from hall_pic_rt.coulomb import collide_ee, coulomb_log
from hall_pic_rt.pusher import boris_push
from hall_pic_rt.walls import see_yield

RESULTS = []


def check(name, ok, detail=""):
    """Zapisuje wynik jednego sprawdzianu i wypisuje go od razu."""
    RESULTS.append((name, ok))
    print(f"[{'OK ' if ok else 'ŹLE'}] {name}   {detail}")


# ---------------------------------------------------------------- Poisson
def test_poisson():
    """Porównuje wyznaczony potencjał z rozwiązaniem podanym wzorem.

    Bierzemy rozkład ładunku będący pojedynczą falą wokół komory i sinusem w
    poprzek. Dla takiego rozkładu potencjał daje się wypisać wzorem, więc mamy z
    czym porównać. Błąd powinien maleć jak kwadrat rozmiaru komórki.
    """
    errs = []
    for N in (32, 64, 128):
        cfg = ConfigRT(N_r=N, N_th=N, n_ppc=1)
        p = PoissonRT(cfg)
        r = cfg.r_nodes[:, None]
        th = cfg.th_nodes[None, :]
        kr = np.pi / cfg.L_r
        kth = 2.0 * np.pi / cfg.L_th
        phi_exact = np.sin(kr * r) * np.cos(kth * th)
        rho = EPS0 * (kr**2 + kth**2) * phi_exact
        phi, _, _ = p.solve(rho, 0.0, 0.0)
        errs.append(np.max(np.abs(phi - phi_exact)) / np.max(np.abs(phi_exact)))
    rate = np.log2(errs[0] / errs[-1]) / 2.0
    check("Poisson: zgodność ze wzorem i rząd dokładności",
          errs[-1] < 2e-3 and rate > 1.7,
          f"błąd {errs[-1]:.2e}, rząd {rate:.2f} (oczekiwany 2)")


# ---------------------------------------------------------------- pchacz
def test_boris_drift():
    """Sprawdza, czy elektron w skrzyżowanych polach dryfuje tam, gdzie powinien.

    Pole magnetyczne leży w poprzek komory, pole elektryczne wzdłuż jej osi.
    Elektron powinien dryfować dokładnie wokół komory z szybkością równą
    ilorazowi obu pól, i nie powinien mieć dryfu w żadnym innym kierunku.
    """
    # Czastka puszczona dokładnie z prędkością dryfu powinna przy niej zostać.
    # Sprawdzamy właśnie to, a nie średnią po torze, bo średnia po niepełnej
    # liczbie obiegów sama z siebie odbiega od dryfu i mierzyłaby nie schemat,
    # lecz długość przebiegu.
    cfg = ConfigRT(N_r=16, N_th=16, n_ppc=1, dt=1e-12)
    v_expect = cfg.E_z / cfg.B_r
    sp = SpeciesRT("e", -E_CHARGE, M_ELECTRON, capacity=10)
    sp.add([cfg.L_r / 2], [cfg.L_th / 2], [0.0], [v_expect], [0.0], [1.0])
    dev = 0.0
    for _ in range(20000):
        boris_push(sp, {"W_axial": (None, None, cfg.E_z)}, cfg)
        sp.x1[0] = cfg.L_r / 2          # trzymamy czastkę w miejscu, badamy samą prędkość
        sp.x2[0] = cfg.L_th / 2
        dev = max(dev, abs(sp.v2[0] - v_expect), abs(sp.v1[0]), abs(sp.v3[0]))
    err = dev / v_expect
    check("Pchacz: dryf wokół komory zgadza się z E/B",
          err < 1e-10, f"odchylenie od dryfu {v_expect:.4e} m/s: {err:.2e}")

    # Energia w samym polu magnetycznym musi być zachowana dokładnie.
    sp2 = SpeciesRT("e", -E_CHARGE, M_ELECTRON, capacity=10)
    sp2.add([cfg.L_r/2], [cfg.L_th/2], [1e6], [2e6], [-5e5], [1.0])
    e0 = float(sp2.kinetic_energy_eV()[0])
    for _ in range(50000):
        boris_push(sp2, {"W_axial": (None, None, 0.0)}, cfg)
        sp2.x1[0] = cfg.L_r / 2
        sp2.x2[0] = cfg.L_th / 2
    e1 = float(sp2.kinetic_energy_eV()[0])
    check("Pchacz: sam obrót w polu magnetycznym nie zmienia energii",
          abs(e1 - e0) / e0 < 1e-12, f"zmiana względna {abs(e1-e0)/e0:.2e}")


# ---------------------------------------------------------------- metryczka
def test_ledger_closure():
    """Sprawdza, czy rachunek energii każdej czastki zgadza się z jej energią.

    Suma energii początkowej i wszystkich zapisanych przyrostów musi równać się
    bieżącej energii czastki. Gdyby jakikolwiek sposób zmiany energii nie był
    zapisywany, ta różnica by odstawała, a przypisanie energii do mechanizmów
    przestałoby cokolwiek znaczyć.
    """
    cfg = ConfigRT(N_r=24, N_th=24, n_ppc=15, enable_mcc=True, coulomb_interval=3,
                   axial_length=1.0e-2, source_mode="controlled")
    sim = SimulationRT(cfg)
    for _ in range(150):
        sim.step_once()
    resid = sim.electrons.ledger_residual()
    eps = sim.electrons.kinetic_energy_eV()
    rel = float(np.max(np.abs(resid)) / np.mean(eps))
    check("Metryczka: rachunek każdej czastki domyka się",
          rel < 1e-9, f"największa reszta {rel:.2e} energii średniej")

    eb = sim.book.summary(sim.electrons, sim.ions)
    check("Rachunek energii: księga całego obszaru domyka się",
          abs(eb["residual_rel"]) < 1e-9, f"reszta względna {eb['residual_rel']:.2e}")


# ---------------------------------------------------------------- zderzenia e-e
def test_coulomb_conservation():
    """Sprawdza, czy zderzenia między elektronami zachowują pęd i energię."""
    cfg = ConfigRT(N_r=8, N_th=8, n_ppc=200, coulomb_interval=1)
    rng = np.random.default_rng(7)
    sp = SpeciesRT("e", -E_CHARGE, M_ELECTRON, capacity=200000, tracked=True)
    n = cfg.n_cells * cfg.n_ppc
    vth = np.sqrt(E_CHARGE * 10.0 / M_ELECTRON)
    sp.add(rng.random(n) * cfg.L_r, rng.random(n) * cfg.L_th,
           rng.normal(0, vth, n), rng.normal(0, vth, n), rng.normal(0, vth, n),
           np.full(n, cfg.w_ref))
    p0 = np.array([np.sum(sp.aw * sp.av1), np.sum(sp.aw * sp.av2), np.sum(sp.aw * sp.av3)])
    e0 = float(np.sum(sp.aw * sp.speed2()))
    for _ in range(50):
        collide_ee(sp, cfg, rng, dt_coll=1e-9)
    p1 = np.array([np.sum(sp.aw * sp.av1), np.sum(sp.aw * sp.av2), np.sum(sp.aw * sp.av3)])
    e1 = float(np.sum(sp.aw * sp.speed2()))
    scale = np.sqrt(e0 * np.sum(sp.aw))
    dp = np.max(np.abs(p1 - p0)) / scale
    de = abs(e1 - e0) / e0
    check("Zderzenia e-e: pęd i energia zachowane",
          dp < 1e-12 and de < 1e-12, f"pęd {dp:.2e}, energia {de:.2e}")


def _relaxation_time(n0, seed=11):
    """Mierzy czas wyrównywania temperatur dwóch grup elektronów.

    Zwraca zmierzony czas oraz czas zderzeń własnych elektronów wg Spitzera dla
    tych samych warunków, żeby dało się porównać jedno z drugim.
    """
    cfg = ConfigRT(N_r=4, N_th=4, n_ppc=4000, coulomb_interval=1, n0=n0)
    rng = np.random.default_rng(seed)
    sp = SpeciesRT("e", -E_CHARGE, M_ELECTRON, capacity=400000, tracked=False)
    n = cfg.n_cells * cfg.n_ppc
    T_hot, T_cold = 20.0, 5.0
    half = n // 2
    x1 = rng.random(n) * cfg.L_r
    x2 = rng.random(n) * cfg.L_th
    v = np.empty((3, n))
    for k in range(3):
        v[k, :half] = rng.normal(0, np.sqrt(E_CHARGE * T_hot / M_ELECTRON), half)
        v[k, half:] = rng.normal(0, np.sqrt(E_CHARGE * T_cold / M_ELECTRON), n - half)
    sp.add(x1, x2, v[0], v[1], v[2], np.full(n, cfg.w_ref))

    def temps():
        """Temperatury obu grup, wyznaczone z rozrzutu prędkości."""
        e = sp.kinetic_energy_eV()
        return (2.0 / 3.0) * e[:half].mean(), (2.0 / 3.0) * e[half:].mean()

    T0h, T0c = temps()
    n_e = cfg.n0
    T_mean = 0.5 * (T_hot + T_cold)
    lnL = float(coulomb_log(np.array([n_e]), np.array([T_mean]), cfg)[0])
    # Czas zderzeń własnych elektronów wg Spitzera, w układzie SI.
    tau_spitzer = (3.0 * np.sqrt(M_ELECTRON) * (E_CHARGE * T_mean)**1.5
                   * (4.0 * np.pi * EPS0)**2
                   / (4.0 * np.sqrt(2.0 * np.pi) * n_e * E_CHARGE**4 * lnL))

    dt_coll = tau_spitzer / 40.0
    n_steps = 120
    dT = []
    for k in range(n_steps):
        collide_ee(sp, cfg, rng, dt_coll=dt_coll)
        th, tc = temps()
        dT.append(th - tc)
    dT = np.array(dT)
    t = (np.arange(n_steps) + 1) * dt_coll
    m = dT > 0.15 * dT[0]
    slope = np.polyfit(t[m], np.log(dT[m]), 1)[0]
    return -1.0 / slope, tau_spitzer


def test_coulomb_relaxation():
    """Porównuje tempo wyrównywania temperatur z wynikiem teoretycznym.

    Startujemy z dwóch grup elektronów o różnych temperaturach w tej samej
    objętości i mierzymy, jak szybko różnica temperatur zanika.

    Sprawdzamy dwie rzeczy. Po pierwsze, czy zmierzony czas leży w skali czasu
    zderzeń własnych elektronów podanej przez Spitzera; sam współczynnik nie
    musi wyjść równy jedności, bo wyrównanie temperatur wymaga kilku zderzeń, a
    definicje czasu zderzeń różnią się między źródłami o czynniki rzędu
    jedności. Po drugie, i to jest mocniejszy sprawdzian, czy zmierzony czas
    zmienia się z gęstością tak, jak powinien: dwukrotnie gęstsza plazma ma
    wyrównywać temperatury mniej więcej dwa razy szybciej. Zgodność samego
    skalowania świadczy o poprawności wzoru w kodzie znacznie lepiej niż
    trafienie w jedną liczbę.
    """
    tau1, sp1 = _relaxation_time(5.0e18)
    ratio = tau1 / sp1
    check("Zderzenia e-e: tempo wyrównywania w skali czasu Spitzera",
          0.3 < ratio < 4.0,
          f"zmierzony {tau1:.3e} s, Spitzer {sp1:.3e} s, iloraz {ratio:.2f}")

    tau2, sp2 = _relaxation_time(2.0e19)
    scal_meas = tau1 / tau2
    scal_theory = sp1 / sp2
    check("Zderzenia e-e: zależność od gęstości zgodna z teorią",
          abs(scal_meas / scal_theory - 1.0) < 0.25,
          f"zmierzone przyspieszenie {scal_meas:.2f}x, teoretyczne {scal_theory:.2f}x")


# ---------------------------------------------------------------- grzanie liczbowe
def test_numerical_heating():
    """Mierzy, jak szybko sam schemat obliczeń podgrzewa plazmę.

    Bierzemy jednorodną plazmę bez pola wzdłuż osi, bez dosyłu, bez zderzeń i z
    odbijającymi ściankami, więc energia nie ma prawa nigdzie uciec ani skądś
    przyjść. Wszystko, co przybędzie, jest błędem obliczeń.

    Ten sprawdzian decyduje o doborze siatki, bo grzanie liczbowe podrabia
    dokładnie to zjawisko, które badamy: rozgrzewa tło i wypycha ogon rozkładu.
    Porównujemy dwa ustawienia różniące się stosunkiem rozmiaru komórki do
    długości ekranowania, przy tej samej liczbie czastek na komórkę. Komórka
    większa od długości ekranowania grzeje szybko i takiej używać nie wolno.

    Mierzymy nachylenie w drugiej połowie przebiegu, a nie zmianę od samego
    początku. Na starcie czastki rozłożone są losowo, więc plazma najpierw
    porządkuje się do stanu równowagi i przy okazji uwalnia trochę energii; to
    jednorazowy skok, a nie grzanie, i wliczanie go zawyżałoby wynik.
    """
    print("   h/λ_D    skok początkowy    grzanie w stanie ustalonym")
    rates = {}
    for L_r, n0 in ((6.4e-3, 5.0e16), (4.8e-3, 2.0e16)):
        cfg = ConfigRT(N_r=36, N_th=64, L_r=L_r, L_th=8.5e-3, n0=n0, n_ppc=40,
                       dt=1.5e-11, E_z=0.0, wall_mode="specular", source_on=False,
                       enable_coulomb_ee=False, enable_mcc=False,
                       track_origin=False, seed=3)
        sim = SimulationRT(cfg)
        ke, ts = [], []
        n_steps = int(3e-7 / cfg.dt)
        for k in range(n_steps):
            sim.step_once()
            if k % 500 == 0:
                ke.append(sim.book.kinetic(sim.electrons) + sim.book.kinetic(sim.ions))
                ts.append(sim.t)
        ke = np.array(ke) / ke[0]
        ts = np.array(ts)
        m = ts > 0.5 * ts[-1]
        rate = float(np.polyfit(ts[m], ke[m], 1)[0]) * 1e-6
        jump = float(np.interp(0.5 * ts[-1], ts, ke) - 1.0)
        ratio = cfg.h_r / cfg.lambda_D
        rates[round(ratio, 2)] = rate
        print(f"   {ratio:5.2f}    {jump*100:+8.2f} %        {rate*100:+8.2f} %/us")

    fine = min(rates)                       # ustawienie z mniejszą komórką
    check("Grzanie liczbowe przy siatce użytej w badaniu jest małe",
          abs(rates[fine]) < 3.0,
          f"przy h/λ_D = {fine} wynosi {rates[fine]*100:+.1f} % energii na mikrosekundę")


# ---------------------------------------------------------------- ścianki
def test_walls():
    """Sprawdza emisję wtórną i szczelność ścianek."""
    cfg = ConfigRT()
    y = see_yield(np.array([0.0, 50.0, 200.0, 1000.0]), cfg)
    check("Ścianki: wydajność emisji wtórnej rośnie i nasyca się",
          abs(y[0] - cfg.see_gamma0) < 1e-12 and y[1] > y[0] and y[-1] <= cfg.see_max + 1e-12,
          f"γ(0)={y[0]:.3f}, γ(50)={y[1]:.3f}, γ(1000)={y[-1]:.3f}")

    # Przy odbijających ściankach nie może zginąć ani jedna czastka.
    cfg2 = ConfigRT(N_r=24, N_th=24, n_ppc=10, wall_mode="specular",
                    source_on=False, enable_coulomb_ee=False)
    sim = SimulationRT(cfg2)
    n0 = sim.electrons.N
    sim.run(300)
    inside = np.all((sim.electrons.ax1 >= 0) & (sim.electrons.ax1 <= cfg2.L_r))
    check("Ścianki odbijające: nic nie ginie i nic nie wychodzi poza obszar",
          sim.electrons.N == n0 and inside,
          f"{n0} -> {sim.electrons.N}")

    # Odbita czastka nie jest stracona, więc sterowany dosył nie ma czego
    # odtwarzać. Okno uśredniania ubytku ustawiamy na jeden krok, a powrót do
    # zadanej gęstości na czas nieskończenie długi: dosył odtwarza wtedy
    # dokładnie ubytek z poprzedniego kroku, więc każda odbita czastka,
    # policzona omyłkowo jako stracona, od razu dałaby nową.
    cfg3 = ConfigRT(N_r=24, N_th=24, n_ppc=10, wall_mode="specular",
                    source_on=True, source_mode="controlled", axial_length=0.0,
                    enable_coulomb_ee=False, source_loss_window=cfg.dt,
                    source_relax_time=1e3)
    sim3 = SimulationRT(cfg3)
    n0 = sim3.electrons.N
    sim3.run(300)
    check("Ścianki odbijające: sterowany dosył nie dokłada czastek",
          sim3.electrons.N == n0, f"{n0} -> {sim3.electrons.N}")


# ---------------------------------------------------------------- rozłożenie pola
def test_axial_loss():
    """Porównuje ubytek wzdłuż osi komory z wynikiem podanym wzorem.

    Bierzemy czastki rozłożone równomiernie wzdłuż osi, o rozkładzie cieplnym
    prędkości, bez żadnych pól i bez zderzeń, więc lecą swobodnie. Dla takiego
    przypadku da się wypisać wzorem, jaka ich część pozostanie po zadanym
    czasie: czastka przetrwa, jeśli jej droga nie wyprowadzi jej poza obszar, a
    dla ustalonej prędkości udział takich położeń początkowych jest wprost
    proporcjonalny do tego, ile z obszaru zostaje.

    Sprawdzenie robimy przy czasie krótszym niż przelot przez obszar. Przy
    dłuższym wzór liniowy przestaje obowiązywać, bo część czastek zdążyłaby
    wyjść niezależnie od położenia początkowego.
    """
    L_z = 4.0e-3
    cfg = ConfigRT(N_r=16, N_th=16, n_ppc=60, dt=1e-11, E_z=0.0, B_r=0.0,
                   wall_mode="specular", source_on=False, enable_coulomb_ee=False,
                   track_origin=False, axial_length=L_z, Te0_eV=10.0, seed=5)
    sim = SimulationRT(cfg)
    N0 = sim.electrons.N
    n_steps = 150
    sim.run(n_steps)
    frac = sim.electrons.N / N0

    t = n_steps * cfg.dt
    vth = np.sqrt(E_CHARGE * cfg.Te0_eV / M_ELECTRON)
    v = np.linspace(-8.0 * vth, 8.0 * vth, 20001)
    pdf = np.exp(-0.5 * (v / vth)**2) / (vth * np.sqrt(2.0 * np.pi))
    survive = np.maximum(1.0 - np.abs(v) * t / L_z, 0.0)
    expected = float(np.trapezoid(pdf * survive, v))
    err = abs(frac - expected) / expected
    check("Ucieczka wzdłuż osi zgadza się ze wzorem",
          err < 0.02, f"zostało {frac:.4f}, ze wzoru {expected:.4f}, błąd {err*100:.2f} %")

    # Przy wyłączonym członie nic nie może ubyć tą drogą.
    cfg2 = ConfigRT(N_r=16, N_th=16, n_ppc=10, wall_mode="specular",
                    source_on=False, enable_coulomb_ee=False, axial_length=0.0)
    sim2 = SimulationRT(cfg2)
    n0 = sim2.electrons.N
    sim2.run(200)
    check("Wyłączona ucieczka osiowa nie zabiera żadnej czastki",
          sim2.electrons.N == n0 and sim2.axial_w == 0.0,
          f"{n0} -> {sim2.electrons.N}")


def test_quasineutrality():
    """Sprawdza, czy plazma pozostaje obojętna elektrycznie i czy gęstość stoi.

    Sprawdzian powstał po tym, jak dosył sterowany ubytkiem samych elektronów
    zaczął, po dodaniu ucieczki wzdłuż osi komory, dokładać jonów w nadmiarze.
    Ucieczka osiowa dotyka bowiem elektronów daleko mocniej niż ciężkich jonów,
    więc pary dosyłane za każdy utracony elektron zostawiały nadmiarowe jony.
    Plazma ładowała się dodatnio, jej potencjał rósł do setek woltów i sam
    rozpędzał elektrony do energii branych potem za wynik fizyczny.

    Najgorsze w tym błędzie było to, że nie naruszał żadnego prawa zachowania:
    księga energii domykała się co do ostatniej cyfry, bo cała energia
    rzeczywiście skądś pochodziła. Wykryć go można było tylko patrząc wprost na
    obojętność plazmy, i po to jest ten sprawdzian.
    """
    cfg = ConfigRT(N_r=32, N_th=48, L_r=3.6e-3, L_th=6.0e-3, n0=3.0e16, n_ppc=24,
                   dt=1.2e-11, source_mode="controlled", axial_length=1.0e-2,
                   wall_mode="see", field_mode="full", seed=9)
    # Przebieg musi być dłuższy niż czas, w jakim sterowanie dosyłem reaguje,
    # inaczej mierzylibyśmy sam stan przejściowy, a nie to, do czego dochodzi.
    sim = SimulationRT(cfg)
    sim.run(20000)
    n_e = sim.electrons.N * cfg.w_ref / cfg.area
    n_i = sim.ions.N * cfg.w_ref / cfg.area
    imbalance = abs(n_e - n_i) / n_i
    drift = abs(n_e - cfg.n0) / cfg.n0
    phi_span = float(sim.phi.max() - sim.phi.min())
    Te = float((2.0 / 3.0) * np.average(sim.electrons.kinetic_energy_eV(),
                                        weights=sim.electrons.aw))
    # Kilka procent nadmiaru ładunku dodatniego to stan normalny plazmy
    # ograniczonej pochłaniającymi ściankami; kilkadziesiąt procent, jak przy
    # opisanym wyżej błędzie, to już nie.
    check("Plazma pozostaje obojętna elektrycznie",
          imbalance < 0.06, f"nierównowaga ładunku {imbalance*100:.2f} %")
    check("Sterowany dosył utrzymuje zadaną gęstość",
          drift < 0.10, f"odchyłka gęstości {drift*100:.2f} %")
    # Rozpiętość potencjału ma być rzędu kilku temperatur elektronowych. Wartość
    # o rząd większa oznacza, że plazma naładowała się jako całość.
    check("Rozpiętość potencjału jest w skali warstwy przyściennej",
          phi_span < 20.0 * Te,
          f"{phi_span:.1f} V przy temperaturze {Te:.1f} eV "
          f"(czyli {phi_span/max(Te,1e-9):.1f} jej krotności)")


def test_field_decomposition():
    """Sprawdza, czy rozłożenie pola na składniki jest zupełne i rozłączne."""
    cfg = ConfigRT(N_r=32, N_th=32, n_ppc=10)
    sim = SimulationRT(cfg)
    for _ in range(60):
        sim.step_once()
    s = sim.E_r_mean + sim.E_r_fluc - sim.E_r
    ok_sum = np.max(np.abs(s)) < 1e-9 * max(np.max(np.abs(sim.E_r)), 1e-30)
    # Część uśredniona nie może zależeć od położenia wokół komory,
    # a zaburzenie musi mieć zerową średnią w tym kierunku.
    ok_mean = np.max(np.abs(sim.E_r_mean - sim.E_r_mean[:, :1])) < 1e-20
    ok_fluc = np.max(np.abs(sim.E_r_fluc.mean(axis=1))) < 1e-9 * max(
        np.max(np.abs(sim.E_r)), 1e-30)
    check("Rozłożenie pola: składniki sumują się do pola i są rozłączne",
          ok_sum and ok_mean and ok_fluc,
          f"suma {np.max(np.abs(s)):.2e}")

    # Przy polu tylko uśrednionym zaburzenia nie mogą wykonywać żadnej pracy.
    cfg2 = ConfigRT(N_r=32, N_th=32, n_ppc=10, field_mode="mean_only")
    sim2 = SimulationRT(cfg2)
    for _ in range(60):
        sim2.step_once()
    eb = sim2.book.summary(sim2.electrons, sim2.ions)
    check("Tryb bez turbulencji: zaburzenia nie wykonują pracy",
          eb["W_turb_th_e"] == 0.0 and eb["W_turb_r_e"] == 0.0,
          f"praca zaburzeń {eb['W_turb_th_e']:.2e} J")


# ---------------------------------------------------------------- ścianka pływająca
def test_floating_wall():
    """Sprawdza ściankę pływającą: warunek brzegowy, zachowanie ładunku, warstwę.

    Trzy rzeczy naraz. Po pierwsze sam warunek brzegowy: przy pustym obszarze i
    równomiernie rozłożonym ładunku powierzchniowym potencjał musi wyjść wszędzie
    równy ilorazowi ładunku przez pojemność warstwy ceramicznej, bo tak zachowuje
    się zwykły kondensator.

    Po drugie zachowanie ładunku. Przy pochłaniających ściankach, bez dosyłu i bez
    ucieczki wzdłuż osi, obszar jest zamknięty: ładunek, który znika z plazmy, musi
    w całości znaleźć się na ściankach. Jest to sprawdzian tego samego rodzaju co
    księga energii i pilnuje tego samego rodzaju błędu, który wywrócił trzecią
    rundę przebiegów.

    Po trzecie sama fizyka: elektrony są ruchliwsze od jonów, więc ścianka musi
    naładować się ujemnie, a plazma ustalić się na potencjale dodatnim względem
    niej, w skali kilku temperatur elektronowych.
    """
    cfg = ConfigRT(N_r=16, N_th=8, wall_bc="floating", wall_eps_r=4.0,
                   wall_thickness=2.0e-3)
    p = PoissonRT(cfg)
    C = EPS0 * cfg.wall_eps_r / cfg.wall_thickness
    sig = np.full(cfg.N_th, -1.0e-7)
    phi, _, _ = p.solve(np.zeros((cfg.n_r_nodes, cfg.n_th_nodes)),
                        sigma_lo=sig, sigma_hi=sig)
    want = -1.0e-7 / C
    err = float(np.max(np.abs(phi - want))) / abs(want)
    check("Ścianka pływająca: potencjał zgadza się z pojemnością ceramiki",
          err < 1e-10, f"{phi.mean():.6f} V wobec {want:.6f} V, błąd {err:.1e}")

    cfg2 = ConfigRT(N_r=24, N_th=24, n_ppc=20, dt=5.0e-12, wall_bc="floating",
                    wall_mode="absorb", source_on=False, enable_coulomb_ee=False,
                    axial_length=0.0, E_z=0.0, seed=7)
    sim = SimulationRT(cfg2)
    q0 = (E_CHARGE * (np.sum(sim.ions.aw) - np.sum(sim.electrons.aw))
          + sim.wall.total_charge())
    sim.run(1500)
    q1 = (E_CHARGE * (np.sum(sim.ions.aw) - np.sum(sim.electrons.aw))
          + sim.wall.total_charge())
    scale = abs(sim.wall.total_charge())
    check("Ścianka pływająca: ładunek plazmy i ścianek razem jest zachowany",
          abs(q1 - q0) < 1e-6 * max(scale, 1e-30),
          f"zmiana {q1 - q0:+.2e} C/m wobec {scale:.2e} C/m zebranych na ściankach")

    sigma_mean = float(sim.wall.sigma.mean())
    drop = float(sim.phi[cfg2.N_r // 2].mean() - sim.phi[0].mean())
    check("Ścianka pływająca: ładuje się ujemnie, a plazma stoi nad nią",
          sigma_mean < 0.0 and 1.0 < drop / cfg2.Te0_eV < 10.0,
          f"ładunek {sigma_mean:.2e} C/m2, spadek {drop:.1f} V "
          f"({drop / cfg2.Te0_eV:.1f} krotności temperatury)")


# ---------------------------------------------------------------- obwód
def test_circuit():
    """Sprawdza, czy pętla sprzężenia z obwodem naprawdę się domyka.

    Bierzemy obwód celowo szybki, o pojemności na tyle małej, by zdążył
    zareagować w liczonym czasie. Wtedy napięcie musi ustalić się tam, gdzie prąd
    pobierany przez plazmę zrównuje się z prądem, który obwód jest w stanie
    dostarczyć, czyli na prostej wyznaczonej przez napięcie zasilacza i opór.
    Sprawdzamy właśnie to, bo jest to jedyna liczba, którą da się wypisać
    niezależnie od samej symulacji.

    Osobno sprawdzamy, że tryb wyłączony niczego nie rusza: pole wzdłuż osi ma
    wtedy pozostać dokładnie takie, jakie zadano.
    """
    base = dict(N_r=16, N_th=24, L_r=3.6e-3, L_th=8.0e-3, n0=3.0e16, n_ppc=12,
                dt=1.2e-11, source_mode="controlled", axial_length=1.0e-2,
                enable_coulomb_ee=False, seed=11, circuit_settle_time=3.0e-8,
                diag_interval=200)

    sim = SimulationRT(ConfigRT(circuit_mode="off", **base))
    sim.run(2000)
    check("Obwód wyłączony nie rusza pola wzdłuż osi",
          sim.E_z_now == sim.cfg.E_z, f"{sim.E_z_now:.4e} V/m")

    R, V_ps = 2000.0, 150.0
    sim = SimulationRT(ConfigRT(circuit_mode="voltage", R_circ=R, V_ps=V_ps,
                                L_circ=1.0e-5, C_circ=2.0e-11, V_C_init=V_ps,
                                **base))
    sim.run(4000)
    V = float(np.mean([r["circ_V"] for r in sim.history[-6:]]))
    I = float(np.mean([r["circ_I_d"] for r in sim.history[-6:]]))
    # Na prostej obciążenia napięcie i prąd wiąże wprost opór gałęzi zasilania.
    V_line = V_ps - R * I
    check("Obwód napięciowy dochodzi do punktu pracy na prostej obciążenia",
          abs(V - V_line) < 0.25 * V_ps,
          f"napięcie {V:.1f} V, z prostej obciążenia {V_line:.1f} V przy {I*1e3:.1f} mA")
    check("Obwód narzuca polu wzdłuż osi napięcie podzielone przez długość",
          abs(sim.E_z_now - sim.circuit.V / sim.cfg.axial_length) < 1e-9,
          f"{sim.E_z_now:.4e} V/m")


# ---------------------------------------------------------------- pełna jonizacja
def test_ionization():
    """Sprawdza, czy plazma utrzymuje się z samej jonizacji gazu.

    Przy pełnej jonizacji sztuczny dosył par jest wyłączony, więc jony biorą się
    wyłącznie ze zderzeń elektronów z ksenonem, a brakujące elektrony wpuszcza
    katoda. Sprawdzamy trzy rzeczy: że jonizacja w ogóle zachodzi, że gęstość
    plazmy trzyma się zadanej i że plazma zostaje obojętna elektrycznie. To
    ostatnie jest tu najważniejsze, bo właśnie złamanie obojętności przez źle
    zbudowany dosył unieważniło trzecią rundę przebiegów.
    """
    cfg = ConfigRT(N_r=24, N_th=32, L_r=3.6e-3, L_th=8.0e-3, n0=3.0e16, n_ppc=16,
                   dt=1.2e-11, axial_length=1.0e-2, plasma_source="ionization",
                   n_neutral=2.0e20, enable_coulomb_ee=False, seed=3,
                   diag_interval=500)
    sim = SimulationRT(cfg)
    sim.run(60000)

    check("Pełna jonizacja: sztuczny dosył par jest wyłączony",
          (not cfg.source_on) and cfg.enable_mcc,
          f"dosył {cfg.source_on}, zderzenia z gazem {cfg.enable_mcc}")

    ratio = np.array([r["N_e"] / max(r["N_i"], 1) for r in sim.history[-10:]])
    n_e = np.array([r["n_e_mean"] for r in sim.history[-10:]])
    check("Pełna jonizacja: plazma powstaje ze zderzeń z gazem",
          sim.n_ioniz_total > 100, f"{sim.n_ioniz_total} jonizacji")
    check("Pełna jonizacja: plazma zostaje obojętna elektrycznie",
          abs(float(ratio.mean()) - 1.0) < 0.06,
          f"stosunek elektronów do jonów {ratio.mean():.4f}")
    check("Pełna jonizacja: gęstość trzyma się zadanej",
          0.5 < float(n_e.mean()) / cfg.n0 < 1.8,
          f"{n_e.mean():.3e} wobec zadanej {cfg.n0:.3e} "
          f"przy gęstości gazu {sim.neutral.n:.2e}")


# ---------------------------------------------------------------- przyspieszenie
def test_kernels_match():
    """Porównuje obie drogi liczenia najgorętszych kawałków kroku.

    Rozkładanie czastek na węzły i pchanie ich napisane są dwa razy: raz na
    całych tablicach, raz jako pętla tłumaczona na kod maszynowy. Druga droga
    istnieje wyłącznie po to, żeby liczyć szybciej, więc musi dawać dokładnie ten
    sam wynik. Sprawdzamy oba jądra osobno, na tym samym stanie początkowym.
    """
    from hall_pic_rt import kernels
    from hall_pic_rt.pusher import charge_density, cic_weights, gather, _accumulate

    if not kernels.HAVE_NUMBA:
        check("Jądra liczące: obie drogi dają ten sam wynik", True,
              "tłumaczenie na kod maszynowy niedostępne, sprawdzian pominięty")
        return

    cfg = ConfigRT(N_r=20, N_th=16, n_ppc=12, dt=1.0e-11, seed=17)
    sim = SimulationRT(cfg)
    sim.run(20)
    sp = sim.electrons

    n_total = cfg.n_r_nodes * cfg.n_th_nodes
    fast = charge_density(sp, cfg)
    idxs, wts = cic_weights(sp, cfg)
    slow = _accumulate(idxs, wts, sp.aw * (sp.charge / (cfg.h_r * cfg.h_th)), n_total)
    err_dep = float(np.max(np.abs(fast - slow))) / float(np.max(np.abs(slow)))
    check("Jądra liczące: rozkład ładunku zgadza się z wersją na tablicach",
          err_dep < 1e-12, f"największa różnica względna {err_dep:.2e}")

    # Ten sam krok pchacza, raz jednym sposobem, raz drugim.
    N = sp.N
    saved = {a: getattr(sp, a)[:N].copy()
             for a in ("x1", "x2", "x3", "v1", "v2", "v3")}
    sums = np.zeros(4)
    W = [np.zeros(N) for _ in range(4)]
    kernels.push_species(
        sp.x1[:N], sp.x2[:N], sp.x3[:N], sp.v1[:N], sp.v2[:N], sp.v3[:N], sp.w[:N],
        sim.E_r_mean.ravel(), sim.E_r_fluc.ravel(), sim.E_th_fluc.ravel(), True,
        sim.E_z_now, cfg.h_r, cfg.h_th, cfg.N_r, cfg.n_th_nodes, cfg.dt,
        sp.charge, sp.mass, E_CHARGE, float(cfg.B_r), cfg.axial_length > 0.0,
        cfg.L_th, W[0], W[1], W[2], W[3], True, sums)
    got = {a: getattr(sp, a)[:N].copy() for a in ("x1", "x2", "v1", "v2", "v3")}

    for a in saved:
        getattr(sp, a)[:N] = saved[a]
    Er_mean_p = gather(sim.E_r_mean, idxs, wts, N)
    parts = {
        "W_axial": (None, None, sim.E_z_now),
        "W_sheath": (Er_mean_p, None, None),
        "W_turb_r": (gather(sim.E_r_fluc, idxs, wts, N), None, None),
        "W_turb_th": (None, gather(sim.E_th_fluc, idxs, wts, N), None),
    }
    led = {}
    boris_push(sp, parts, cfg, ledger_out=led)
    sp.x2[:N] = np.mod(sp.x2[:N], cfg.L_th)

    worst = 0.0
    for a in ("x1", "x2", "v1", "v2", "v3"):
        ref = getattr(sp, a)[:N]
        scale = max(float(np.max(np.abs(ref))), 1e-300)
        worst = max(worst, float(np.max(np.abs(got[a] - ref))) / scale)
    led_err = 0.0
    for name, arr in zip(("W_axial", "W_sheath", "W_turb_r", "W_turb_th"), W):
        ref = led[name]
        scale = max(float(np.max(np.abs(ref))), 1e-300)
        led_err = max(led_err, float(np.max(np.abs(arr - ref))) / scale)
    check("Jądra liczące: pchacz zgadza się z wersją na tablicach",
          worst < 1e-12 and led_err < 1e-11,
          f"położenia i prędkości {worst:.2e}, rozdział pracy pola {led_err:.2e}")


def test_ion_subcycle():
    """Sprawdza rzadsze pchanie jonów: kiedy wolno, kiedy nie, i czy jest wierne.

    Jony są kilkaset tysięcy razy cięższe od elektronów i w ciągu jednego kroku
    prawie się nie ruszają, więc kusi, żeby pchać je rzadziej. Pierwsza wersja
    tego sprawdzianu tak właśnie orzekła i była w błędzie: porównywała przebiegi
    w wycinku o połowę węższym wokół komory, w którym niestabilność nie zdążyła
    się rozwinąć. Przy nastawach badania rzadsze pchanie zaniża temperaturę tła
    o blisko połowę, bo jony przestają nadążać nie za własnym ruchem, lecz za
    falą: wzór zaburzenia jest niesiony dryfem elektronów i przechodzi przez
    komórkę szybciej, niż trwa wydłużony krok jonowy.

    Dlatego sprawdzamy teraz trzy rzeczy. Że warunek na krok jonowy odrzuca
    nastawy, przy których zmierzyliśmy błąd. Że tam, gdzie warunek jest
    spełniony, wynik naprawdę się nie zmienia. I że rozkład ładunku jonów,
    utrzymywany między pchnięciami przyrostowo, zgadza się z tablicami czastek,
    bo niepełna poprawka też dawała temperaturę dwa razy za wysoką.
    """
    from hall_pic_rt.pusher import charge_density

    # Nastawy badania. Wzór fali przechodzi przy nich przez komórkę na tyle
    # szybko, że dziesięciokrotny krok jonowy musi zostać odrzucony.
    study = dict(N_r=36, N_th=80, L_r=3.6e-3, L_th=8.0e-3, n0=3.0e16, n_ppc=32,
                 dt=1.2e-11, source_mode="controlled", axial_length=1.0e-2)
    ok_guard = False
    try:
        ConfigRT(ion_subcycle=10, **study)
    except ValueError:
        ok_guard = True
    c1 = ConfigRT(ion_subcycle=1, **study)
    check("Rzadsze pchanie jonów: warunek odrzuca nastawy, przy których zawodzi",
          ok_guard,
          f"przy kroku pojedynczym wzór fali przechodzi "
          f"{c1.ion_step_over_wave:.2f} komórki, przy dziesięciokrotnym "
          f"{10*c1.ion_step_over_wave:.2f}, a wolno {c1.ion_subcycle_limit:.2f}")

    # Nastawy, przy których warunek jest spełniony: słabszy napęd, więc wolniejszy
    # dryf i wolniej przesuwający się wzór fali.
    base = dict(N_r=36, N_th=40, L_r=3.6e-3, L_th=4.0e-3, n0=3.0e16, n_ppc=24,
                dt=1.2e-11, E_z=2.0e3, source_mode="controlled",
                axial_length=1.0e-2, enable_coulomb_ee=False,
                diag_interval=250, seed=23)

    sim = SimulationRT(ConfigRT(ion_subcycle=10, **base))
    worst = 0.0
    for _ in range(37):
        sim.step_once()
        fresh = charge_density(sim.ions, sim.cfg)
        scale = max(float(np.max(np.abs(fresh))), 1e-300)
        worst = max(worst, float(np.max(np.abs(sim._rho_i - fresh))) / scale)
    check("Rzadsze pchanie jonów: rozkład ładunku zgadza się z tablicami czastek",
          worst < 1e-12, f"największa różnica względna {worst:.2e}")

    out = {}
    for k in (1, 10):
        sim = SimulationRT(ConfigRT(ion_subcycle=k, **base))
        sim.run(15000)
        eps = sim.electrons.kinetic_energy_eV()
        out[k] = ((2.0 / 3.0) * float(np.mean(eps)),
                  sim.electrons.N * sim.cfg.w_ref / sim.cfg.area,
                  sim.ions.N * sim.cfg.w_ref / sim.cfg.area)
    dTe = abs(out[10][0] - out[1][0]) / out[1][0]
    dne = abs(out[10][1] - out[1][1]) / out[1][1]
    dni = abs(out[10][2] - out[1][2]) / out[1][2]
    check("Rzadsze pchanie jonów tam, gdzie wolno, nie zmienia wyniku",
          dTe < 0.05 and dne < 0.02 and dni < 0.02,
          f"temperatura {out[1][0]:.2f} wobec {out[10][0]:.2f} eV ({dTe*100:.1f} %), "
          f"gęstości {dne*100:.1f} % i {dni*100:.1f} %")


def main():
    """Uruchamia wszystkie sprawdziany i podsumowuje wynik."""
    print("=" * 70)
    print(" Sprawdziany wersji w poprzek i wokół komory")
    print("=" * 70)
    test_poisson()
    test_boris_drift()
    test_field_decomposition()
    test_axial_loss()
    test_quasineutrality()
    test_ledger_closure()
    test_coulomb_conservation()
    test_coulomb_relaxation()
    test_walls()
    test_floating_wall()
    test_circuit()
    test_ionization()
    test_kernels_match()
    test_ion_subcycle()
    test_numerical_heating()
    print("=" * 70)
    bad = [n for n, ok in RESULTS if not ok]
    print(f" zdane: {len(RESULTS)-len(bad)}/{len(RESULTS)}")
    for n in bad:
        print(f"   niezdane: {n}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
