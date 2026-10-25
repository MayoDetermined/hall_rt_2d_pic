"""Podtrzymanie plazmy: dosył par elektron-jon.

Ścianki bez przerwy pochłaniają czastki, więc bez dosyłu plazma w liczonym
obszarze szybko by zgasła i badanie ogona rozkładu straciłoby sens. Zamiast
liczyć pełną jonizację gazu, dosyłamy pary elektron-jon ze stałą wydajnością na
jednostkę objętości, rozłożone równomiernie po całym obszarze. Odpowiada to
jonizacji zachodzącej gdzie indziej wzdłuż osi komory i wnoszonej do liczonego
wycinka razem z przepływem.
"""

import numpy as np

from hall_pic_rt.constants import E_CHARGE, M_ELECTRON
from .species import ORIG_SOURCE, ORIG_CATHODE


class NeutralGas:
    """Gaz roboczy, z którego bierze się plazma przy pełnej jonizacji.

    Sterowanie jest celowo wolne, wolniejsze od czasu przebywania czastki w
    obszarze, żeby nie mieszało się do interesujacych zjawisk"""

    def __init__(self, cfg):
        """Zapamiętuje gęstość początkową i wartość, do której dążymy."""
        
        self.cfg = cfg
        self.n = float(cfg.n_neutral)
        self.n_target_plasma = cfg.source_target_n
        self.consumed = 0.0          # ile atomów zjonizowano od początku przebiegu
        self._alpha = min(1.0, cfg.dt / max(cfg.source_loss_window, cfg.dt))
        self._loss = 0.0             # uśredniony ubytek jonów na krok
        self._prod = 0.0             # uśredniona produkcja jonów na krok

    def update(self, lost_w_i, made_w_i, n_e_now, dt):
        """Dostraja gęstość gazu tak, by jonizacja nadążała za ubytkiem.

        Sterujemy dwiema rzeczami naraz. Pierwsza jest ważniejsza to ze jony powstają
        wyłącznie z jonizacji, więc w stanie ustalonym musi ich powstawać tyle,
        ile ubywa. Iloraz jednego przez drugie mówi wtedy, ile razy za mało albo
        za dużo gazu mamy, bo tempo jonizacji jest wprost proporcjonalne do jego
        gęstości. 

        Druga to zwykła poprawka na odchyłkę gęstości plazmy od zadanej, żeby
        wszystkie porównywane przebiegi trzymały tę samą gęstość.

        Oba człony razem ograniczamy, żeby pojedynczy nierówny krok nie szastal
        gazem o rzedy wielkości.
        """
        cfg = self.cfg
        self._loss += self._alpha * (lost_w_i - self._loss)
        self._prod += self._alpha * (made_w_i - self._prod)

        if cfg.neutral_mode != "controlled":
            return self.n

        if self._prod > 0.0:
            balance = float(np.log(max(self._loss, 1e-30) / self._prod))
        
        else:
            # Jonizacji nie ma wcale, więc gaz musi gęstnieć najszybciej, jak wolno.
            balance = 3.0
        err_n = 0.0
        
        if self.n_target_plasma > 0.0:
            err_n = (self.n_target_plasma - n_e_now) / self.n_target_plasma
        
        drive = float(np.clip(balance + err_n, -3.0, 3.0))
        
        self.n *= float(np.exp(dt / cfg.neutral_relax_time * drive))
        self.n = float(np.clip(self.n, cfg.neutral_min, cfg.neutral_max))
        
        return self.n


class CathodeSource:
    """Prąd elektronów wpuszczany do kanału, odpowiednik katody silnika.

    Wpuszczamy tyle elektronów, by ładunek zgromadzony w
    obszarze razem z ładunkiem zebranym na ściankach schodził do zera w czasie
    krótszym niż czas przebywania czastki, ale dłuższym niż krok.
    """

    def __init__(self, cfg, rng):
        """Zapamiętuje nastawy i zeruje resztę przenoszoną między krokami."""
        self.cfg = cfg
        self.rng = rng
        self._rem = 0.0
        self._alpha = min(1.0, cfg.dt / max(cfg.source_loss_window, cfg.dt))
        self._loss = 0.0             # uśredniony ubytek elektronów na krok
        self._prod = 0.0             # uśredniona produkcja elektronów z jonizacji
        self.w_last = 0.0

    def inject(self, electrons, lost_w_e, made_w_e, q_net, t_now):
        """Wpuszcza elektrony pokrywające to, czego jonizacja nie odtworzy.
        """
        cfg, rng = self.cfg, self.rng
        
        self.w_last = 0.0
        self._loss += self._alpha * (lost_w_e - self._loss)
        self._prod += self._alpha * (made_w_e - self._prod)
        
        if not cfg.cathode_on:
            return 0.0, 0.0

        w_need = self._loss - self._prod
        
        if q_net > 0.0:
            w_need += (q_net / E_CHARGE) * (cfg.dt / cfg.cathode_relax_time)
        
        want = w_need / cfg.w_ref + self._rem
        n = int(np.floor(max(want, 0.0)))
        
        self._rem = max(want, 0.0) - n
        
        if n <= 0:
            return 0.0, 0.0

        x1 = rng.random(n) * cfg.L_r
        x2 = rng.random(n) * cfg.L_th
        x3 = rng.random(n) * cfg.axial_length if cfg.axial_length > 0 else np.zeros(n)
        
        vth = np.sqrt(E_CHARGE * cfg.cathode_Te_eV / M_ELECTRON)
        v = [rng.normal(0.0, vth, n) for _ in range(3)]
        w = np.full(n, cfg.w_ref)
        
        electrons.add(x1, x2, v[0], v[1], v[2], w,
                      orig=ORIG_CATHODE, t_now=t_now, x3=x3)
        self.w_last = float(np.sum(w))
        E_in = float(np.sum(w * 0.5 * M_ELECTRON * sum(u**2 for u in v) / E_CHARGE))
        return self.w_last, E_in


class PairSource:
    """Dosyła pary elektron-jon, utrzymując plazmę mimo strat na ściankach."""

    def __init__(self, cfg, rng):
        """Zapamiętuje wydajność dosyłu i przelicza ją na wagę na jeden krok."""
        
        self.cfg = cfg
        self.rng = rng
        
        # Waga par dosyłanych w jednym kroku, liczona na jednostkę długości osi.
        self.w_per_step = cfg.source_rate * cfg.area * cfg.dt
        self._remainder = 0.0
        
        # Uśredniony ubytek na ściankach przypadający na jeden krok. Startuje od
        # wydajności zadanej z góry, żeby sterowanie miało od czego zacząć.
        self._loss_avg = {"e": self.w_per_step, "i": self.w_per_step}
        self._alpha = min(1.0, cfg.dt / max(cfg.source_loss_window, cfg.dt))
       
        self.w_target = cfg.source_target_n * cfg.area
        self.rate_now = cfg.source_rate
        
        self._rem = {"pair": 0.0, "e": 0.0}

    def _controlled_weight(self, sp, lost_weight, which):
        """Wylicza, ile wagi jednego gatunku dosłać w tym kroku.

        Składa się to z dwóch części. Pierwsza odtwarza to, co w ostatnim czasie
        ubyło, uśrednione po oknie dłuższym niż pojedynczy krok, bo ubytek w
        jednym kroku jest bardzo nierówny. Druga powoli ściąga gęstość do
        wartości zadanelj, jeśli zdążyła od niej odejść.
        """
        cfg = self.cfg
        self._loss_avg[which] += self._alpha * (lost_weight - self._loss_avg[which])
        w_now = sp.N * cfg.w_ref
        
        correction = (self.w_target - w_now) * cfg.dt / cfg.source_relax_time
        return max(self._loss_avg[which] + correction, 0.0)

    def _count(self, w_step, key):
        """Zamienia zamówioną wagę na całkowitą liczbę czastek modelowych.

        Reszta, która nie wypadła całkowita, przechodzi na następny krok, dzięki
        czemu wydajność dosyłu jest dokładna w dłuższym czasie, a nie tylko
        średnio.
        """
        want = w_step / self.cfg.w_ref + self._rem[key]
        n = int(np.floor(max(want, 0.0)))
        self._rem[key] = max(want, 0.0) - n
        return n

    def _positions(self, n):
        """Losuje położenia n nowych czastek, równomiernie w całym obszarze."""
        cfg, rng = self.cfg, self.rng
        x1 = rng.random(n) * cfg.L_r
        x2 = rng.random(n) * cfg.L_th
        x3 = rng.random(n) * cfg.axial_length if cfg.axial_length > 0 else np.zeros(n)
        return x1, x2, x3

    def inject(self, electrons, ions, t_now, lost_w_e=0.0, lost_w_i=0.0):
        """Dosyła czastki tak, by gęstość stała i by plazma została obojętna.

        Dosył ma dwie części, bo w silniku mają je dwa różne zjawiska. Jonizacja
        tworzy pary elektron-jon i to ona odtwarza ubytek jonów. Elektronów ubywa
        jednak szybciej, bo uchodzą także wzdłuż osi komory, a tej różnicy nie da
        się pokryć parami, bo dokładałaby jonów, których nikt nie zabiera. Brakujące
        elektrony dosyłamy więc osobno
        """
        cfg, rng = self.cfg, self.rng
        if not cfg.source_on:
            return 0.0, 0.0, 0.0

        if cfg.source_mode == "controlled":
            want_e = self._controlled_weight(electrons, lost_w_e, "e")
            want_i = self._controlled_weight(ions, lost_w_i, "i")
        else:
            want_e = want_i = self.w_per_step
        self.rate_now = want_i / (cfg.area * cfg.dt)

        n_pair = self._count(min(want_e, want_i), "pair")
        n_e_only = self._count(max(want_e - want_i, 0.0), "e")
        if n_pair <= 0 and n_e_only <= 0:
            return 0.0, 0.0, 0.0

        vth_e = np.sqrt(E_CHARGE * cfg.source_Te_eV / M_ELECTRON)
        vth_i = np.sqrt(E_CHARGE * cfg.source_Ti_eV / cfg.m_ion)
        w_tot = E_e = E_i = 0.0

        if n_pair > 0:
            x1, x2, x3 = self._positions(n_pair)
            
            w = np.full(n_pair, cfg.w_ref)
            ve = [rng.normal(0.0, vth_e, n_pair) for _ in range(3)]
            vi = [rng.normal(0.0, vth_i, n_pair) for _ in range(3)]
            
            electrons.add(x1, x2, ve[0], ve[1], ve[2], w,
                          orig=ORIG_SOURCE, t_now=t_now, x3=x3)
            
            ions.add(x1.copy(), x2.copy(), vi[0], vi[1], vi[2], w.copy(), x3=x3.copy())
            
            w_tot += float(np.sum(w))
            
            E_e += float(np.sum(w * 0.5 * M_ELECTRON * sum(v**2 for v in ve) / E_CHARGE))
            E_i += float(np.sum(w * 0.5 * cfg.m_ion * sum(v**2 for v in vi) / E_CHARGE))

        if n_e_only > 0:
            x1, x2, x3 = self._positions(n_e_only)
            
            w = np.full(n_e_only, cfg.w_ref)
            ve = [rng.normal(0.0, vth_e, n_e_only) for _ in range(3)]
            
            electrons.add(x1, x2, ve[0], ve[1], ve[2], w,
                          orig=ORIG_SOURCE, t_now=t_now, x3=x3)
            w_tot += float(np.sum(w))
            E_e += float(np.sum(w * 0.5 * M_ELECTRON * sum(v**2 for v in ve) / E_CHARGE))

        return w_tot, E_e, E_i
