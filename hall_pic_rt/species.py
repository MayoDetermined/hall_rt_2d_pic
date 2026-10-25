"""Pojemnik na czastki modelowe wraz z metryczką pochodzenia.

Czastka ma dwie współrzędne położenia, w poprzek i wokół komory, oraz pełne trzy
składowe prędkości. trzecia biegnie wzdłuż osi komory, której położenia nie
liczymy. Poza tym każdy elektron niesie własną metryczkę: skąd się wziął, kiedy,
w którym miejscu, ile razy zetknął się ze ścianką oraz osobny rachunek energii
mówiący, która składowa pola i który mechanizm ile mu jej dołożyły.

Metryczka jest tym, co pozwala odpowiedzieć na pytanie o pochodzenie szybkich
elektronów.
"""

import numpy as np

from hall_pic_rt.constants import E_CHARGE


# Skąd wzięła się czastka. Numery trzymamy w tablicy, żeby dało się je liczyć
# tak samo szybko jak resztę.
ORIG_SEED = 0        # rozstawiona na starcie
ORIG_SOURCE = 1      # dosłana przez podtrzymanie plazmy
ORIG_WALL = 2        # wybita ze ścianki jako elektron wtórny
ORIG_IONIZ = 3       # powstała przy jonizacji gazu
ORIG_CATHODE = 4     # wpuszczona przez katodę

ORIGIN_NAMES = {
    ORIG_SEED: "start",
    ORIG_SOURCE: "dosył",
    ORIG_WALL: "ścianka",
    ORIG_IONIZ: "jonizacja",
    ORIG_CATHODE: "katoda",
}

# Składniki rachunku energii pojedynczej czastki. Pierwsze cztery to praca
# poszczególnych składowych pola elektrycznego, ostatnie dwa to energia dodana
# poza polem.
LEDGER_FIELDS = ("W_axial", "W_sheath", "W_turb_th", "W_turb_r", "W_coll", "W_ee")

LEDGER_LABELS = {
    "W_axial": "pole wzdłuż osi",
    "W_sheath": "warstwa przyścienna",
    "W_turb_th": "zaburzenie wokół komory",
    "W_turb_r": "zaburzenie w poprzek",
    "W_coll": "zderzenia z gazem",
    "W_ee": "zderzenia elektron-elektron",
}


class SpeciesRT:
    """Zbiór czastek modelowych jednego gatunku, z opcjonalną metryczką pochodzenia."""

    # Trzecia współrzędna położenia biegnie wzdłuż osi komory. Nie ma jej na
    # siatce i nie bierze udziału w wyznaczaniu pola, bo płaszczyzna obliczeń
    # jej nie obejmuje. Służy wyłącznie do rozstrzygnięcia, czy czastka zdążyła
    # już opuścić obszar wzdłuż osi.
    _BASE = ("x1", "x2", "x3", "v1", "v2", "v3", "w")
    _TAGS = ("t_birth", "x1_birth", "E_birth") + LEDGER_FIELDS
    _INTS = ("orig", "n_wall")

    def __init__(self, name, charge, mass, capacity=200000, tracked=False):
        """Tworzy pusty gatunek. Metryczkę prowadzimy tylko tam, gdzie ma sens."""
        self.name = name
        self.charge = charge
        
        self.mass = mass
        self.tracked = tracked
        
        self.N = 0
        self._cap = int(capacity)
        
        for a in self._BASE:
            setattr(self, a, np.zeros(self._cap))
        
        if tracked:
            for a in self._TAGS:
                setattr(self, a, np.zeros(self._cap))
            
            for a in self._INTS:
                setattr(self, a, np.zeros(self._cap, dtype=np.int32))

    @property
    def _fields(self):
        """Nazwy wszystkich tablic czastek, zależnie od tego, czy prowadzimy metryczke."""
        if self.tracked:
            return self._BASE + self._TAGS + self._INTS
        return self._BASE

    def _ensure(self, extra):
        """Powiększa tablice, gdy nowe czastki się nie mieszczą."""
        need = self.N + extra
        if need <= self._cap:
            return
        
        newcap = max(need, int(self._cap * 2))
        for a in self._fields:
            arr = getattr(self, a)
            new = np.zeros(newcap, dtype=arr.dtype)
            new[:self.N] = arr[:self.N]
            setattr(self, a, new)
        
        self._cap = newcap

    def add(self, x1, x2, v1, v2, v3, w, orig=None, t_now=0.0, x3=0.0):
        """Dopisuje nowe czastki na końcu tablic i zakłada im metryczkę.

        Metryczka nowej czastki startuje od zera we wszystkich składnikach
        rachunku energii, a jej energia początkowa zapisywana jest osobno.
        Dzięki temu suma energii początkowej i wszystkich składników powinna w
        każdej chwili równać się bieżącej energii czastki.
        """
        x1 = np.atleast_1d(np.asarray(x1, dtype=float))
        n = x1.size
        if n == 0:
            return
        
        self._ensure(n)
        
        s = slice(self.N, self.N + n)
        
        self.x1[s] = x1
        self.x2[s] = np.atleast_1d(x2)
        self.x3[s] = np.atleast_1d(x3)
        
        self.v1[s] = np.atleast_1d(v1)
        self.v2[s] = np.atleast_1d(v2)
        self.v3[s] = np.atleast_1d(v3)
        
        self.w[s] = np.atleast_1d(w)
        
        if self.tracked:
            for a in self._TAGS:
                getattr(self, a)[s] = 0.0
            
            self.t_birth[s] = t_now
            self.x1_birth[s] = x1
            
            self.E_birth[s] = 0.5 * self.mass * (
                np.atleast_1d(v1)**2 + np.atleast_1d(v2)**2 + np.atleast_1d(v3)**2) / E_CHARGE
           
            self.orig[s] = ORIG_SEED if orig is None else orig
            self.n_wall[s] = 0
        self.N += n

    def remove_mask(self, kill, stable=False):
        """Usuwa zaznaczone czastki i upycha pozostałe na początek tablic.
        """
        dead = np.flatnonzero(kill)
        
        if dead.size == 0:
            return
        m = self.N - dead.size

        if stable:
            idx = np.flatnonzero(~kill)
            for a in self._fields:
                arr = getattr(self, a)
                arr[:m] = arr[idx]
            self.N = m
            return

        # Dziury poniżej nowego końca zatykamy żywymi czastkami zza niego.
        holes = dead[dead < m]
        if holes.size:
            tail = np.flatnonzero(~kill[m:self.N]) + m
            for a in self._fields:
                arr = getattr(self, a)
                arr[holes] = arr[tail]
        self.N = m

    # ---- widoki aktywnych czastek ----
    @property
    def ax1(self):
        """Położenia w poprzek komory."""
        return self.x1[:self.N]

    @property
    def ax2(self):
        """Położenia wokół komory."""
        return self.x2[:self.N]

    @property
    def ax3(self):
        """Położenia wzdłuż osi komory, poza płaszczyzną obliczeń."""
        return self.x3[:self.N]

    @property
    def av1(self):
        """Prędkości w poprzek komory."""
        return self.v1[:self.N]

    @property
    def av2(self):
        """Prędkości wokół komory."""
        return self.v2[:self.N]

    @property
    def av3(self):
        """Prędkości wzdłuż osi komory, czyli poza płaszczyzną obliczeń."""
        return self.v3[:self.N]

    @property
    def aw(self):
        """Wagi aktywnych czastek."""
        return self.w[:self.N]

    def speed2(self):
        """Kwadrat szybkości każdej aktywnej czastki."""
        return self.av1**2 + self.av2**2 + self.av3**2

    def kinetic_energy_eV(self):
        """Energia ruchu każdej aktywnej czastki, w elektronowoltach."""
        return 0.5 * self.mass * self.speed2() / E_CHARGE

    # ---- metryczka ----
    def ledger(self):
        """Zwraca rachunek energii aktywnych czastek jako słownik tablic, w eV."""
        if not self.tracked:
            return {}
        return {a: getattr(self, a)[:self.N] for a in LEDGER_FIELDS}

    def ledger_sum(self):
        """Suma wszystkich składników rachunku energii, dla każdej czastki, w eV."""
        if not self.tracked:
            return np.zeros(self.N)
        tot = np.zeros(self.N)
        for a in LEDGER_FIELDS:
            tot += getattr(self, a)[:self.N]
        return tot

    def ledger_residual(self):
        """Różnica między bieżącą energią a sumą metryczki, w eV.
        """
        if not self.tracked or self.N == 0:
            return np.zeros(0)
        return self.kinetic_energy_eV() - (self.E_birth[:self.N] + self.ledger_sum())
