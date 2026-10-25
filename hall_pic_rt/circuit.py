"""Zewnętrzny obwód wyznaczający pole elektryczne wzdłuż osi komory.

Prąd wyprowadzamy z tego, ile ładunku i w którą
stronę opuszcza obszar wzdłuż osi komory, napięcie liczymy z równań obwodu, a
pole wzdłuż osi jest ilorazem napięcia i długości obszaru przyspieszania.

Obwód działa w dwóch odmianach. a) to zasilacz o zadanym napięciu, podany
przez opór i cewkę, z kondensatorem wiszącym równolegle do wyładowania b) 
 to zasilacz o zadanym prądzie, z doborem napięcia tak, by
prąd wyładowania trzymał się tam zadanej wartości
"""

import numpy as np


class DischargeCircuit:
    """Obwód zasilający z narzuacjacym napieciem """

    def __init__(self, cfg):
        """wartości elementów i stan początkowy z konfiguracji """
        self.cfg = cfg
        self.mode = cfg.circuit_mode
        self.V_ps = cfg.V_ps
        self.R = cfg.R_circ
        self.Lc = cfg.L_circ
        self.C = cfg.C_circ
        self.I_set = cfg.I_set
        self.I_L = cfg.I_L_init

        #  trzeba spojrzec na znak napięcia by zbadać kondesator
        if cfg.V_C_init < 0.0:
            self.V = cfg.V_ps if self.mode == "voltage" else cfg.E_z * cfg.axial_length
        else:
            self.V = cfg.V_C_init
            
        self.I_d = 0.0            # uśredniony prąd wyładowania
        self.I_raw = 0.0          # prąd zmierzony w ostatnim kroku

        self._alpha = min(1.0, cfg.dt / max(cfg.circuit_current_window, cfg.dt))
        # Zadany prąd niepodany oznacza, że wezmiemy za zadany ten, ktory sam
        # popłynie po czasie ustalania.
        self._auto_set = cfg.I_set <= 0.0
        self._t = 0.0

    # ---- prąd pobierany przez plazmę ----
    def discharge_current(self, q_up, q_down, dt):
        """Zamienia ładunek, który odszedł wzdłuż osi, na prąd wyładowania.

        Podajemy ładunek, który opuścił obszar przez stronę anodową i przez
        stronę wylotową, liczony na jednostkę długości osi komory. Elektron
        odchodzący ku anodzie i jon odchodzący ku wylotowi dają prąd tego samego
        znaku, bo prąd umowny płynie w obu przypadkach ku wylotowi.

        Wynik przeliczamy z liczonego wycinka na cały pierścień kanału, bo tyle
        widzi obwód.
        """
        return self.cfg.circuit_scale * (q_down - q_up) / dt

    # ---- całkowanie obwodu ----
    def _deriv(self, I_L, V, I_d):
        """Zwraca tempo zmian prądu w cewce i napięcia na kondensatorze."""
        if self.mode == "current":
            # Zasilacz o zadanym prądzie: nadmiar prądu ładuje kondensator, a
            # niedobór go rozładowuje, więc napięcie samo dochodzi tam, gdzie
            # prąd wyładowania zrównuje się z zadanym.
            return 0.0, (self.I_set - I_d) / self.C
        
        dI = (self.V_ps - self.R * I_L - V) / self.Lc
        dV = (I_L - I_d) / self.C
        
        return dI, dV

    def advance(self, I_d, dt):
        """Przesuwa stan obwodu o jeden krok przy zadanym prądzie wyładowania.

        Prąd wyładowania w obrębie kroku traktujemy jako stały, a sam krok
        dzielimy na kilka mniejszych i każdy liczymy metodą Rungego-Kutty
        czwartego rzędu. Zwraca napięcie, które narzuca polu wzdłuż osi.
        """
        self.I_raw = I_d

        # Uśredniamy, bo w pojedynczym kroku jest duzo szumu 
        # pojedynczych czastek, a obwód widzi cały pierścień kanału naraz.
        self.I_d += self._alpha * (I_d - self.I_d)
        I_d = self.I_d
        self._t += dt
        
        if self._auto_set and self._t >= self.cfg.circuit_settle_time:
            self.I_set = self.I_d
            self._auto_set = False
        
        if self.mode == "off":
            return self.V
        
        if self.mode == "current" and self._auto_set:
            # Dopóki nie wiadomo, jaki prąd trzymać, nie ruszamy napięcia.
            return self.V

        n = max(int(self.cfg.circuit_substeps), 1)
        h = dt / n
        I_L, V = self.I_L, self.V

        for _ in range(n):
            k1 = self._deriv(I_L, V, I_d)
            k2 = self._deriv(I_L + 0.5*h*k1[0], V + 0.5*h*k1[1], I_d)
            k3 = self._deriv(I_L + 0.5*h*k2[0], V + 0.5*h*k2[1], I_d)
            k4 = self._deriv(I_L + h*k3[0], V + h*k3[1], I_d)
            I_L += (h/6.0) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
            V += (h/6.0) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        
        if not np.isfinite(V):
            V = self.V
        
        if not np.isfinite(I_L):
            I_L = self.I_L
        
        # Napięcie ujemne oznaczałoby, że pole zawraca jony do anody. Model
        # tego nie ogarnia, więc zatrzymujemy się na zerze i odznaczamy 
        # jako stan, w którym obwód wyszedł poza swój zakres ważności.
        self.V = float(np.clip(V, 0.0, self.cfg.V_max))
        self.I_L = float(I_L)
        return self.V

    def E_z(self):
        """Pole elektryczne wzdłuż osi komory wynikające z napięcia na wyładowaniu."""
        if self.mode == "off":
            return self.cfg.E_z
        return self.V / self.cfg.axial_length

    def state(self):
        """Zwraca stan obwodu do zapisania w szeregu czasowym."""
        return {"circ_V": self.V, "circ_I_L": self.I_L,
                "circ_I_d": self.I_d, "circ_E_z": self.E_z()}
