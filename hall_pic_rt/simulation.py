"""Główna pętla symulacji na płaszczyźnie w poprzek i wokół komory.

Jeden krok wygląda nastepujaco. Rozkładamy ładunek na siatkę i wyznaczamy z niego
potencjał oraz pole. Pole od razu rozkładamy na część uśrednioną wokół komory,
czyli warstwę przyścienną, i na zaburzenia, czyli turbulencję. Odczytujemy oba
kawałki osobno w miejscu każdej czastki, dokładamy zadane z góry pole wzdłuż osi
komory i przesuwamy czastki, prowadząc przy tym osobny rachunek pracy każdej
składowej. Potem obsługujemy ścianki, dosyłamy plazmę, rozgrywamy zderzenia z
gazem i między elektronami, na koniec zbieramy diagnostykę.
"""

import numpy as np

from hall_pic_rt.constants import E_CHARGE, M_ELECTRON
from .config import ConfigRT
from .species import SpeciesRT, LEDGER_FIELDS, ORIG_SEED
from .poisson import PoissonRT
from .pusher import (cic_weights, deposit_charge, deposit_number_density,
                     deposit_moment, gather, boris_push, cell_index,
                     charge_density, add_charge_of_new, add_charge_of_mask)
from .walls import (wall_events, axial_events, emit_secondaries, WallSurface)
from .source import PairSource, NeutralGas, CathodeSource
from .circuit import DischargeCircuit
from .coulomb import collide_ee
from .mcc import NeutralMCC
from .energetics import EnergyBook, FIELD_TERMS
from . import kernels
from . import origin

# Kolejność, w jakiej jądro liczące oddaje sumy pracy poszczególnych składowych
# pola. Musi zgadzać się z kolejnością w samym jądrze.
FIELD_TERMS_ORDER = ("W_axial", "W_sheath", "W_turb_r", "W_turb_th")

# Pusty wynik obsługi brzegów, używany w tych krokach, w których jony stoją.
_EMPTY_WALL = {"w_lo": 0.0, "w_hi": 0.0, "E_out_eV": 0.0, "E_see_eV": 0.0,
               "w_see": 0.0, "n_see": 0, "q_wall": 0.0,
               "ledger_lost": {a: 0.0 for a in LEDGER_FIELDS},
               "E_birth_lost": 0.0, "kill": None, "emit": None,
               "eps_hit": None, "x2_hit": None, "side_hit": None, "w_hit": None}
_EMPTY_AXIAL = {"w": 0.0, "E_eV": 0.0, "ledger": {a: 0.0 for a in LEDGER_FIELDS},
                "E_birth": 0.0, "n": 0, "kill": None, "q_up": 0.0, "q_down": 0.0}


class SimulationRT:
    """Prowadzi przebieg na płaszczyźnie w poprzek i wokół komory."""

    def __init__(self, cfg: ConfigRT):
        """Przygotowuje pola, czastki, ścianki, dosył, zderzenia i księgę energii."""
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.t = 0.0
        self.step = 0

        cap = max(200000, 2 * cfg.n_cells * cfg.n_ppc)
        self.electrons = SpeciesRT("e", -E_CHARGE, M_ELECTRON, capacity=cap,
                                   tracked=cfg.track_origin)
        self.ions = SpeciesRT("Xe+", +E_CHARGE, cfg.m_ion, capacity=cap, tracked=False)

        self.poisson = PoissonRT(cfg)
        self.source = PairSource(cfg, self.rng)
        self.mcc = NeutralMCC(cfg, self.rng) if cfg.enable_mcc else None
        self.book = EnergyBook(cfg)

        # Ścianka pływająca zbiera ładunek; przy ściance o zadanym potencjale
        # ładunek odpływa do masy i nie ma czego zbierać.
        self.wall = WallSurface(cfg) if cfg.wall_bc == "floating" else None

        # Obwód zewnętrzny. W trybie wyłączonym oddaje po prostu zadane pole.
        self.circuit = DischargeCircuit(cfg)
        self.E_z_now = self.circuit.E_z()

        # Gaz roboczy i katoda mają sens tylko wtedy, gdy plazma bierze się z
        # jonizacji, a nie z dosyłu gotowych par.
        ioniz = cfg.plasma_source == "ionization"

        self.neutral = NeutralGas(cfg) if ioniz else None
        self.cathode = CathodeSource(cfg, self.rng) if ioniz else None
        
        self.w_cathode = 0.0

        # Pamięć na rozkład ładunku jonów między kolejnymi ich pchnięciami oraz
        # na sumy pracy pola, żeby nie zakładać ich w każdym kroku od nowa.
        self._rho_i = None
        self._n_i_dep = 0

        self._sums = np.zeros(4)

        self._q_ion_rate = 0.0
        self._made_w_last = 0.0
        kernels.warmup()

        shape = (cfg.n_r_nodes, cfg.n_th_nodes)
        self.phi = np.zeros(shape)
        self.E_r = np.zeros(shape)
        self.E_th = np.zeros(shape)

        self.E_r_mean = np.zeros(shape)
        self.E_r_fluc = np.zeros(shape)
        self.E_th_fluc = np.zeros(shape)

        # Wielkości zbiorcze narastające w czasie.
        self.wall_w = {"e_lo": 0.0, "e_hi": 0.0, "i_lo": 0.0, "i_hi": 0.0}
        self.wall_w_see = 0.0
        self.n_see_total = 0
        self.n_ioniz_total = 0

        # Metryczka czastek, które zginęły na ściankach: dokąd trafiła energia
        # i skąd wcześniej pochodziła.
        self.lost_ledger = {a: 0.0 for a in LEDGER_FIELDS}
        self.lost_E_birth = 0.0

        # To samo dla czastek, które odeszły wzdłuż osi komory.
        self.axial_w = 0.0
        self.axial_w_i = 0.0
        self.axial_ledger = {a: 0.0 for a in LEDGER_FIELDS}

        self.history = []
        self.snapshots = []
        self._snap_todo = sorted(cfg.snapshot_times)

        self._seed_plasma()
        self.book.set_initial(self.electrons, self.ions)

    # ---- plazma początkowa ----
    def _seed_plasma(self):
        """Rozstawia plazmę startową elektrony i jony w tych samych miejscach.

        Wspólne położenia sprawiają, że plazma zaczyna od stanu obojętnego, więc
        nie startujemy od sztucznego pola, co by bylo slabe
        """
        cfg, rng = self.cfg, self.rng
        N = cfg.n_cells * cfg.n_ppc
        cell = np.repeat(np.arange(cfg.n_cells), cfg.n_ppc)

        i = cell // cfg.N_th
        j = cell % cfg.N_th

        x1 = (i + rng.random(N)) * cfg.h_r
        x2 = (j + rng.random(N)) * cfg.h_th

        # Wzdłuż osi komory rozstawiamy czastki równomiernie, bo plazma jest w
        # tym kierunku jednorodna.

        x3 = rng.random(N) * cfg.axial_length if cfg.axial_length > 0 else 0.0

        vth_e = np.sqrt(E_CHARGE * cfg.Te0_eV / M_ELECTRON)
        vth_i = np.sqrt(E_CHARGE * cfg.Ti0_eV / cfg.m_ion)

        w = np.full(N, cfg.w_ref)

        self.electrons.add(x1, x2,
                           rng.normal(0, vth_e, N), rng.normal(0, vth_e, N),
                           rng.normal(0, vth_e, N), w,
                           orig=ORIG_SEED, t_now=0.0, x3=x3)
        
        self.ions.add(x1.copy(), x2.copy(),
                      rng.normal(0, vth_i, N), rng.normal(0, vth_i, N),
                      rng.normal(0, vth_i, N), w.copy(),
                      x3=x3 if np.isscalar(x3) else x3.copy())

    # ---- pojedynczy krok ----
    def step_once(self):
        """Wykonuje jeden pełny krok w czasie."""

        cfg = self.cfg

        # Jony są kilkaset tysięcy razy cięższe od elektronów, więc w ciągu
        # jednego kroku prawie się nie ruszają. Pchamy je rzadziej, za to od
        # razu o odpowiednio dłuższy czas. ich ładunek rozkładamy na siatkę co
        # krok, bo potencjał musi go widzieć zawsze.
        push_ions = (self.step % cfg.ion_subcycle == 0)

        rho = self._charge_density()

        if self.wall is not None:
            self.phi, self.E_r, self.E_th = self.poisson.solve(
                rho, sigma_lo=self.wall.sigma[0], sigma_hi=self.wall.sigma[1])
            
        else:
            self.phi, self.E_r, self.E_th = self.poisson.solve(
                rho, cfg.wall_phi, cfg.wall_phi)

        # Rozłożenie pola na warstwę przyścienną i na zaburzenia.
        self.E_r_mean, self.E_r_fluc, self.E_th_fluc, _ = self.poisson.decompose(
            self.E_r, self.E_th)

        self._push_species(self.electrons, "e", cfg.dt)
       
        if push_ions:
            self._push_species(self.ions, "i", cfg.dt * cfg.ion_subcycle)
            # Jony właśnie się przesunęły, więc ich rozkład ładunku trzeba
            # policzyć od nowa. Od tej chwili aż do następnego pchnięcia
            # utrzymujemy go przyrostowo: odejmujemy te, które giną, i
            # dokładamy te, które powstają.
            self._rho_i = charge_density(self.ions, cfg)

        # Ścianki, kierunek zawijający się wokół komory oraz ucieczka wzdłuż
        # osi. Oba ubytki rozstrzygamy naraz i dopiero potem raz upychamy
        # tablice, bo przepisywanie ich dwa razy byłoby dwa razy droższe.
        res_e, ax_e = self._boundaries(self.electrons, allow_see=True)
        if push_ions:
            res_i, ax_i = self._boundaries(self.ions, allow_see=False)
        else:
            res_i, ax_i = _EMPTY_WALL, _EMPTY_AXIAL
        self._record_walls(res_e, res_i)
        self._record_axial(ax_e, ax_i)

        # Obwód zewnętrzny. Prąd wyładowania bierze się z ładunku, który odszedł
        # wzdłuż osi komory, a wynikające z niego napięcie narzuca pole wzdłuż
        # tej osi w następnym kroku.
        self._advance_circuit(ax_e, ax_i, push_ions)

        # Podtrzymanie plazmy. Sterowanemu dosyłowi podajemy ubytek elektronów
        # netto, czyli to, co ścianki zabrały, pomniejszone o to, co oddały jako
        # elektrony wtórne.
        # Ubytek liczymy osobno dla obu gatunków, bo ucieczka wzdłuż osi dotyka
        # ich zupełnie inaczej.
        loss_e = res_e["w_lo"] + res_e["w_hi"] - res_e["w_see"] + ax_e["w"]
        loss_i = res_i["w_lo"] + res_i["w_hi"] + ax_i["w"]
        n_i_before = self.ions.N
        w_src, E_src_e, E_src_i = self.source.inject(
            self.electrons, self.ions, self.t, lost_w_e=loss_e, lost_w_i=loss_i)
        self.book.add("E_source_e", E_src_e)
        self.book.add("E_source_i", E_src_i)

        # Zderzenia z gazem. Przy pełnej jonizacji to one, i tylko one, tworzą
        # nową plazmę, więc gęstość gazu dostrajamy tuż przed nimi.

        made_w = 0.0
        if self.mcc is not None:
            if self.neutral is not None:
                n_e_now = self.electrons.N * cfg.w_ref / cfg.area

                # Gęstość gazu dostrajamy z bilansu jonów: powstaje ich tylko z
                # jonizacji, więc musi ich powstawać tyle, ile ubywa. Bierzemy
                # produkcję z poprzedniego kroku, bo tegoroczna dopiero powstanie.
                self.mcc.set_density(self.neutral.update(
                    loss_i, self._made_w_last, n_e_now, cfg.dt))
            
            dE, n_iz = self.mcc.collide(self.electrons, self.ions, self.t)
            
            self.book.add("E_mcc", dE)
            self.n_ioniz_total += n_iz
            
            made_w = n_iz * cfg.w_ref
            self._made_w_last = made_w
            
            if self.neutral is not None:
                self.neutral.consumed += made_w

        # Katoda. Uzupełnia elektrony, których jonizacja nie odtworzy, bo
        # uchodzą wzdłuż osi szybciej niż jony.

        if self.cathode is not None:
            w_cat, E_cat = self.cathode.inject(
                self.electrons, loss_e, made_w, self._net_charge(), self.t)
            
            self.w_cathode += w_cat
            self.book.add("E_source_e", E_cat)

        # Jony dosłane albo powstałe z jonizacji trzeba dołożyć do rozkładu
        # ładunku, bo między pchnięciami liczymy go przyrostowo.
        if self.ions.N > n_i_before and self._rho_i is not None:
            add_charge_of_new(self.ions, cfg, n_i_before, self._rho_i)

        # Zderzenia kulombowskie między elektronami.
        if cfg.enable_coulomb_ee and (self.step % cfg.coulomb_interval == 0):
            dE = collide_ee(self.electrons, cfg, self.rng,
                            dt_coll=cfg.dt * cfg.coulomb_interval)
            self.book.add("E_ee", dE)

        self.t += cfg.dt
        self.step += 1

    # ---- części kroku ----
    def _charge_density(self):
        """Składa rozkład ładunku obu gatunków na węzłach siatki.

        Elektrony rozkładamy co krok od nowa. Jony stoją między swoimi
        pchnięciami, więc ich rozkład jest utrzymywany przyrostowo: przeliczany
        w całości zaraz po każdym pchnięciu, a w międzyczasie poprawiany o te
        czastki, które zginęły albo powstały.
        """
        cfg = self.cfg
        
        if self._rho_i is None:
            self._rho_i = charge_density(self.ions, cfg)
        
        rho_flat = charge_density(self.electrons, cfg)
        rho_flat += self._rho_i
        rho = rho_flat.reshape(cfg.n_r_nodes, cfg.n_th_nodes)
        
        # Węzły na ściankach obejmują pół komórki, więc gęstość w nich jest
        # dwa razy większa, niż wyszłoby z podzielenia przez pełną komórkę.
        rho[0, :] *= 2.0
        rho[-1, :] *= 2.0
        
        return rho

    def _push_species(self, sp, tag, dt):
        """Przesuwa jeden gatunek i dopisuje pracę każdej składowej pola do księgi."""
        if sp.N == 0:
            return
        
        cfg = self.cfg
        N = sp.N
        use_fluc = cfg.field_mode == "full"
        b1 = cfg.B_vector(sp.ax1)[0]

        if kernels.HAVE_NUMBA and np.isscalar(b1):
            if sp.tracked:
                W = (sp.W_axial[:N], sp.W_sheath[:N], sp.W_turb_r[:N], sp.W_turb_th[:N])
            
            else:
                z = np.zeros(1)
                W = (z, z, z, z)
            
            kernels.push_species(
                sp.x1[:N], sp.x2[:N], sp.x3[:N], sp.v1[:N], sp.v2[:N], sp.v3[:N],
                sp.w[:N],
                
                self.E_r_mean.ravel(), self.E_r_fluc.ravel(), self.E_th_fluc.ravel(),
                use_fluc, self.E_z_now, cfg.h_r, cfg.h_th, cfg.N_r, cfg.n_th_nodes,
                
                dt, sp.charge, sp.mass, E_CHARGE, float(b1),
                cfg.axial_length > 0.0, cfg.L_th,
                
                W[0], W[1], W[2], W[3], bool(sp.tracked), self._sums)
            
            for name, val in zip(FIELD_TERMS_ORDER, self._sums):
                self.book.terms[f"{name}_{tag}"] += float(val) * E_CHARGE * self.book.f
            
            return

        # Droga zapasowa na samych tablicach, gdy tłumaczenie na kod maszynowy
        # jest niedostępne
        idxs, wts = cic_weights(sp, cfg)
        Er_mean_p = gather(self.E_r_mean, idxs, wts, N)
        
        if use_fluc:
            parts = {
                "W_axial": (None, None, self.E_z_now),
                "W_sheath": (Er_mean_p, None, None),
                "W_turb_r": (gather(self.E_r_fluc, idxs, wts, N), None, None),
                "W_turb_th": (None, gather(self.E_th_fluc, idxs, wts, N), None),
            }
        else:
            parts = {
                "W_axial": (None, None, self.E_z_now),
                "W_sheath": (Er_mean_p, None, None),
                "W_turb_r": (None, None, None),
                "W_turb_th": (None, None, None),
            }
        led = {}
        w_now = sp.aw.copy()
        boris_push(sp, parts, cfg, ledger_out=led, dt=dt)
        if sp.tracked:
            for name, val in led.items():
                if np.isscalar(val):
                    continue
                getattr(sp, name)[:N] += val
        self.book.add_field_work(tag, led, w_now)

    def _boundaries(self, sp, allow_see):
        """Obsługuje ścianki i ucieczkę osiową jednego gatunku, usuwając raz."""
        cfg = self.cfg
        
        # Pusher zawinął już kierunek wokół komory, więc obsługa ścianek nie
        # musi przechodzić po tablicy jeszcze raz tylko po to samo.
        res = wall_events(sp, cfg, self.rng, self.t, allow_see=allow_see,
                          wrapped=kernels.HAVE_NUMBA)
        
        kill = res["kill"]
        
        ax = axial_events(sp, cfg, skip=kill)

        # Ładunek czastek pochłoniętych zostaje na powierzchni ścianki.
        if self.wall is not None and res["w_hit"] is not None:
            self.wall.collect(res["side_hit"], res["x2_hit"], sp.charge * res["w_hit"])

        if kill is None:
            total = ax["kill"]
        
        elif ax["kill"] is None:
            total = kill
        
        else:
            total = kill | ax["kill"]
        
        if total is not None:
            # Jony pchamy rzadziej, więc ich rozkład ładunku żyje dłużej niż
            # jeden krok. te, które właśnie giną, trzeba z niego odjąć, zanim
            # znikną z tablic.
            if sp is self.ions and self._rho_i is not None:
                add_charge_of_mask(sp, cfg, total, self._rho_i, sign=-1.0)
            
            sp.remove_mask(total, stable=cfg.stable_compact)

        if res["emit"] is not None:
            payload = res["emit"]
            
            emit_secondaries(sp, cfg, payload, self.rng, self.t, out=res)
            
            # Elektron wtórny zabiera ze ścianki ładunek ujemny, więc ścianka
            # robi się od tego bardziej dodatnia.
            if self.wall is not None:
                self.wall.collect(payload["side"], payload["x2"],
                                  -sp.charge * payload["w"])
        return res, ax

    def _net_charge(self):
        """Zwraca nadmiar ładunku w obszarze razem z tym, co zebrały ścianki.

        Liczba ta powinna trzymać się blisko zera, ze wzgledu na quasi obojetnosc
        """
        q = E_CHARGE * (float(np.sum(self.ions.aw)) - float(np.sum(self.electrons.aw)))
        if self.wall is not None:

            q += self.wall.total_charge()
        
        return q

    def _advance_circuit(self, ax_e, ax_i, push_ions):
        """Wylicza prąd wyładowania i przesuwa obwód o jeden krok."""
        cfg = self.cfg
        
        if push_ions and cfg.ion_subcycle > 1:
            # Jony odchodzą skokowo, raz na kilka kroków, więc ich udział w
            # prądzie rozkładamy równo na te kroki. Inaczej obwód widziałby
            # ostre skoki, których w rzeczywistości nie ma.
            
            self._q_ion_rate = ((ax_i["q_down"] - ax_i["q_up"])
                                / (cfg.dt * cfg.ion_subcycle))
        
        elif cfg.ion_subcycle == 1:
            self._q_ion_rate = (ax_i["q_down"] - ax_i["q_up"]) / cfg.dt

        rate = (ax_e["q_down"] - ax_e["q_up"]) / cfg.dt + self._q_ion_rate
        I_d = cfg.circuit_scale * rate
        
        self.circuit.advance(I_d, cfg.dt)
        self.E_z_now = self.circuit.E_z()

    def _record_axial(self, ax_e, ax_i):
        """Zapisuje, ile wagi i energii odeszło wzdłuż osi komory."""
        self.axial_w += ax_e["w"]
        self.axial_w_i += ax_i["w"]
        
        self.book.add("E_axial_e", ax_e["E_eV"])
        self.book.add("E_axial_i", ax_i["E_eV"])
        
        for a in LEDGER_FIELDS:
            self.axial_ledger[a] += ax_e["ledger"][a]

    def _record_walls(self, res_e, res_i):
        """Zapisuje wszystko, co wydarzyło się na ściankach w tym kroku."""
        self.wall_w["e_lo"] += res_e["w_lo"]
        self.wall_w["e_hi"] += res_e["w_hi"]
        
        self.wall_w["i_lo"] += res_i["w_lo"]
        self.wall_w["i_hi"] += res_i["w_hi"]
        
        self.wall_w_see += res_e["w_see"]
        self.n_see_total += res_e["n_see"]
        
        self.book.add("E_wall_e", res_e["E_out_eV"])
        self.book.add("E_wall_i", res_i["E_out_eV"])
        self.book.add("E_see", res_e["E_see_eV"])
        
        for a in LEDGER_FIELDS:
            self.lost_ledger[a] += res_e["ledger_lost"][a]
        self.lost_E_birth += res_e["E_birth_lost"]

    # ---- diagnostyka ----
    def turbulence_measures(self):
        """Mierzy siłę zaburzeń wokół komory i ich rozkład na fale.

        Miarą siły jest średnia kwadratowa pola wokół komory oraz względna
        zmienność gęstości. Rozkład na fale mówi, która długość fali niesie
        najwięcej, co pozwala rozpoznać, czy zaburzenie jest niestabilnością o
        określonej skali, czy tylko szumem liczbowym.
        """
        cfg = self.cfg
        E_th_rms = float(np.sqrt(np.mean(self.E_th_fluc**2)))
        
        E_r_fluc_rms = float(np.sqrt(np.mean(self.E_r_fluc**2)))
        E_r_mean_max = float(np.max(np.abs(self.E_r_mean)))

        ne = deposit_number_density(self.electrons, cfg)
        ne_mean = ne.mean(axis=1, keepdims=True)
        
        dn = ne - ne_mean
        dn_rel = float(np.sqrt(np.mean((dn / np.maximum(ne_mean, 1e-300))**2)))

        # Rozkład zaburzeń na fale biegnące wokół komory. Sama średnia
        # kwadratowa nie wystarcza, bo skończona liczba czastek modelowych sama
        # z siebie daje zaburzenia o zbliżonej wielkości. Szum rozkłada się
        # jednak równo na wszystkie długości fali, a niestabilność skupia się na
        # jednej. Dlatego miarą turbulencji jest tu wysokość szczytu widma
        # ponad tłem, a nie sama jego wysokość.
        spec_n = np.abs(np.fft.rfft(dn, axis=1))**2
        spec_E = np.abs(np.fft.rfft(self.E_th_fluc, axis=1))**2
        
        out = {"E_th_rms": E_th_rms, "E_r_fluc_rms": E_r_fluc_rms,
               "E_r_mean_max": E_r_mean_max, "dn_rel": dn_rel}
        
        for tag, spec in (("n", spec_n), ("E", spec_E)):
            s = spec.mean(axis=0) / self._noise_shape(spec.shape[1], tag)
            s[0] = 0.0
            n_m = s.size
            
            floor = float(np.median(s[max(2, n_m // 2):]))   # tło z krótkich fal
            
            k_peak = int(np.argmax(s[1:n_m // 2]) + 1)
            
            out[f"{tag}_k_peak"] = k_peak
            out[f"{tag}_lambda_mm"] = (cfg.L_th / k_peak) * 1e3 if k_peak > 0 else np.inf
            out[f"{tag}_peak_over_floor"] = float(s[k_peak]) / max(floor, 1e-300)
            
            # Moc skupiona w falach długich, czyli tam, gdzie mieści się
            # niestabilność, po odjęciu tła rozłożonego równo.
            band = slice(1, min(17, n_m // 2))
            n_band = band.stop - band.start
            
            out[f"{tag}_low_k_power"] = float(np.sum(s[band]) - n_band * floor)
            out[f"{tag}_floor"] = floor
        
        out["k_peak"] = out["n_k_peak"]
        out["lambda_peak_mm"] = out["n_lambda_mm"]
        
        # mowi, czy turbulencja wgl się rozwinęła.
        out["turb_coherence"] = out["n_peak_over_floor"]
        return out

    def _noise_shape(self, n_modes, tag):
        """Zwraca kształt widma, jaki daje sam szum skończonej liczby czastek.
        """
        key = (n_modes, tag)
        cached = getattr(self, "_noise_cache", None)
        
        if cached is None:
            cached = self._noise_cache = {}
        
        if key in cached:
            return cached[key]
        m = np.arange(n_modes)
        x = np.pi * m / self.cfg.N_th
        
        sinc = np.ones_like(x)
        sinc[1:] = np.sin(x[1:]) / x[1:]
        shape = sinc**4                      # tłumienie przy rozkładaniu na węzły
        
        if tag == "E":
            # Pole wyznaczane z ładunku wnosi jeszcze czynnik odwrotny do
            # liczby falowej, liczony tak jak w schemacie różnicowym.
            kh = 2.0 * np.pi * m / self.cfg.N_th
            resp = np.ones_like(kh)
            
            with np.errstate(divide="ignore", invalid="ignore"):
                resp[1:] = (np.sin(kh[1:]) / (2.0 - 2.0 * np.cos(kh[1:])))**2
            shape = shape * resp
        
        shape[0] = 1.0
        cached[key] = np.maximum(shape, 1e-30)
        return cached[key]

    def radial_profiles(self):
        """Zwraca profile w poprzek komory, uśrednione wokół niej."""
        cfg = self.cfg
        cache = cic_weights(self.electrons, cfg)
        
        ne = deposit_number_density(self.electrons, cfg, cached=cache)
        eps = self.electrons.kinetic_energy_eV()
        
        e_dens = deposit_moment(self.electrons, cfg, eps, cached=cache)
        fast = (eps > cfg.E_fast_eV).astype(float)
        
        n_fast = deposit_moment(self.electrons, cfg, fast, cached=cache)
        ni = deposit_number_density(self.ions, cfg)

        ne_r = ne.mean(axis=1)
        return {
            "r": cfg.r_nodes,
            "n_e": ne_r,
            "n_i": ni.mean(axis=1),
            "phi": self.phi.mean(axis=1),
            "E_r_mean": self.E_r_mean[:, 0].copy(),
            "Te": (2.0 / 3.0) * e_dens.mean(axis=1) / np.maximum(ne_r, 1e-300),
            "f_fast": n_fast.mean(axis=1) / np.maximum(ne_r, 1e-300),
        }

    def collect(self):
        """Zbiera jeden punkt szeregu czasowego z wszystkimi miarami naraz."""
        cfg = self.cfg
        U = self.poisson.field_energy(self.E_r, self.E_th)
        eb = self.book.summary(self.electrons, self.ions, U_field=U)
        turb = self.turbulence_measures()
        res = origin.summarize(self.electrons, cfg)

        rec = {
            "t": self.t, "step": self.step,
            "N_e": self.electrons.N, "N_i": self.ions.N,
            "n_e_mean": self.electrons.N * cfg.w_ref / cfg.area if self.electrons.N else 0.0,
            "wall_e": self.wall_w["e_lo"] + self.wall_w["e_hi"],
            "wall_i": self.wall_w["i_lo"] + self.wall_w["i_hi"],
            "w_see": self.wall_w_see,
            "q_net": self._net_charge(),
            "w_cathode": self.w_cathode,
            "n_neutral": self.neutral.n if self.neutral is not None else 0.0,
            "n_ioniz": self.n_ioniz_total,
        }
        rec.update(self.circuit.state())
        
        if self.wall is not None:
            rec.update(self.wall.measures(self.phi))
        rec.update({k: v for k, v in turb.items()})
        rec.update({k: v for k, v in eb.items() if not isinstance(v, dict)})
        
        for k, v in res.items():
            if isinstance(v, (int, float, np.floating)):
                rec[k] = float(v)
        af = res.get("attr_fast", {})
        
        for a in LEDGER_FIELDS:
            rec["fast_frac_" + a] = af.get("frac_" + a, np.nan)
            rec["fast_eV_" + a] = af.get(a + "_per_particle", np.nan)
        
        rec["fast_wall_born"] = af.get("frac_wall_born", np.nan)
        rec["fast_touched_wall"] = af.get("frac_touched_wall", np.nan)
        dom = res.get("dom_fast", {})
        
        for a in LEDGER_FIELDS:
            rec["fast_dom_" + a] = dom.get("dom_" + a, np.nan)
        self.history.append(rec)
        return rec

    def snapshot(self):
        """Zapisuje pełny obraz chwili: mapy, profile i rozkład energii."""
        cfg = self.cfg
        
        eps = self.electrons.kinetic_energy_eV()
        bins = np.linspace(0.0, max(4.0 * cfg.E_fast_eV, float(eps.max()) if eps.size else 100.0), 160)
        centers, f = origin.eedf(eps, self.electrons.aw, bins)
        
        snap = {
            "t": self.t,
            "phi": self.phi.copy(),
            "E_th_fluc": self.E_th_fluc.copy(),
            "n_e": deposit_number_density(self.electrons, cfg),
            "profiles": self.radial_profiles(),
            "eedf_eps": centers, "eedf_f": f,
            "report": origin.summarize(self.electrons, cfg),
        }
        
        self.snapshots.append(snap)
        return snap

    def run(self, n_steps=None, progress=None):
        """Prowadzi przebieg, zbierając diagnostykę i zapisując wazne chwile."""
        cfg = self.cfg
        
        n_steps = cfg.n_steps if n_steps is None else n_steps
        
        for _ in range(n_steps):
            self.step_once()
            
            if self._snap_todo and self.t >= self._snap_todo[0]:
                self._snap_todo.pop(0)
                self.snapshot()
           
            if self.step % cfg.diag_interval == 0:
                rec = self.collect()
                if progress is not None:
                    progress(self, rec)
        return self.history

    # ---- opis stanu ----
    def stats_text(self):
        """Składa krótki opis bieżącego stanu, do wypisania albo na wykres."""
        cfg = self.cfg
        
        turb = self.turbulence_measures()
        res = origin.summarize(self.electrons, cfg)
        
        U = self.poisson.field_energy(self.E_r, self.E_th)
        eb = self.book.summary(self.electrons, self.ions, U_field=U)
        
        lines = [
            f"t            : {self.t*1e9:8.1f} ns   krok {self.step}",
            f"N_e / N_i    : {self.electrons.N} / {self.ions.N}",
            f"ścianki      : {cfg.wall_mode}    pole: {cfg.field_mode}",
            f"e-e Coulomb  : {'tak' if cfg.enable_coulomb_ee else 'nie'}",
            "---- turbulencja ----",
            f"E_th rms     : {turb['E_th_rms']:9.1f} V/m",
            f"dn/n rms     : {turb['dn_rel']:9.4f}",
            f"fala szczyt. : m = {turb['k_peak']}, lambda = {turb['lambda_peak_mm']:.2f} mm",
            "---- warstwa przyścienna ----",
            f"|E_r| śr.max : {turb['E_r_mean_max']:9.1f} V/m",
            f"strata e/i   : {self.wall_w['e_lo']+self.wall_w['e_hi']:.3e} / "
            f"{self.wall_w['i_lo']+self.wall_w['i_hi']:.3e}",
            f"wtórne       : {self.wall_w_see:.3e}",
            "---- energia [J] ----",
        ]
        
        for t in FIELD_TERMS:
            lines.append(f"{t:<12s} : {eb[t+'_e']:10.3e}  ({eb['frac_'+t+'_e']*100:5.1f} %)")
        
        lines.append(f"{'ścianki':<12s} : {-eb['E_wall_e']:10.3e}")
        lines.append(f"{'wtórne':<12s} : {eb['E_see']:10.3e}")
        lines.append(f"{'e-e':<12s} : {eb['E_ee']:10.3e}")
        lines.append(f"{'reszta':<12s} : {eb['residual_rel']:10.2e}")
        lines.append("---- szybkie elektrony ----")
        lines.append(origin.format_report(res, cfg))
        
        return "\n".join(lines)
