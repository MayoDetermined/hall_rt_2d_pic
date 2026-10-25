"""Potencjał i pole elektryczne w płaszczyźnie w poprzek i wokół komory.

Wokół komory kierunek zawija się sam w sobie, więc rozkładamy go na niezależne
fale dla każdej z nich zostaje wtedy proste zadanie w poprzek komory, które
rozwiązujemy od razu dla wszystkich fal naraz. Osczedzamy operacje ze względu 
na niskie zuzycie tablic

W poprzek komory ścianki dają się opisać na dwa sposoby, wybierane setingsem.

Pierwszy to ścianka utrzymywana na zadanym potencjale. Co jest oczywscie sporym
uproszczeniem bo raczej ceramiki sie nie podlacza.

Drugi to ścianka pływająca. Ceramika o zadanej grubości i przenikalności leży na
uziemionym korpusie silnika, a na jej powierzchni gromadzi się ładunek zebrany z
plazmy. Potencjał powierzchni nie jest wtedy zadany, tylko wynika z tego, ile
ładunku już się zebrało, i zmienia się wzdłuż obwodu komory.

I wtedy warunek brzegowy bedzie wiazac nachylenie potencjalu po stronie plazmy, potencjał
samej powierzchni oraz zgromadzony ładunek. Ważne jest, że wiazany je liniowo, a
pojemność warstwy ceramicznej jest ta sama na całym obwodzie, więc rozkład na
fale wokół komory nadal rozprzęga zadanie.

Poza samym polem wyznaczamy tu jego rozłożenie na składniki. Pole uśrednione
wokół komory opisuje warstwę przyścienną, czyli to, co robią ścianki, a reszta
opisuje zaburzenia, czyli turbulencję.
"""

import numpy as np

from hall_pic_rt.constants import EPS0


class PoissonRT:
    """Wyznacza potencjał i pole oraz rozkłada pole na część uśrednioną i zaburzenie."""

    def __init__(self, cfg):
        """Przygotowuje rozkład na fale wokół komory i stałe zadania w poprzek."""
        self.cfg = cfg
        self.N_r = cfg.N_r
        self.N_th = cfg.N_th
        self.h_r = cfg.h_r
        self.h_th = cfg.h_th
        self.floating = getattr(cfg, "wall_bc", "fixed") == "floating"

        # Pojemność warstwy ceramicznej przypadająca na jednostkę powierzchni. Granica
        # nieskończonej pojemności odpowiada ściance uziemionej.
        if self.floating:
            self.C_wall = EPS0 * cfg.wall_eps_r / cfg.wall_thickness
        else:
            self.C_wall = np.inf

        # Fale wokół komory. Kierunek jest okresowy, a rozkład ładunku
        # rzeczywisty, więc wystarczy połowa fal a  reszta jest ich odbiciem.
        k = np.arange(self.N_th // 2 + 1)
       
        self.lam = (2.0 * np.cos(2.0 * np.pi * k / self.N_th) - 2.0) / self.h_th**2
        self.n_modes = self.lam.size

        if self.floating:
            # Niewiadomymi są wszystkie węzły, razem z obiema ściankami.
            self.n_int = self.N_r + 1
            self.i0 = 0
        
        else:
            # Potencjał ścianek jest zadany, więc niewiadome są tylko wewnątrz.
            self.n_int = self.N_r - 1
            self.i0 = 1

        self._build_rows()
        self._factorize()

    # ---- układ równań w poprzek komory ----
    def _build_rows(self):
        """Składa trzy przekątne zadania w poprzek komory dla każdej fali.

        Wewnątrz obszaru jest to zwykły trzypunktowy wzór na drugą pochodną.
        Na ściankach, gdy są pływające, węzeł obejmuje tylko pół komórki i
        zamiast wartości potencjału znamy tam bilans ładunku, więc wiersz
        wygląda inaczej: nachylenie po stronie plazmy zestawione jest z
        potencjałem powierzchni przez pojemność warstwy ceramicznej.
        """
        n, h = self.n_int, self.h_r
        inv_h2 = 1.0 / h**2

        a = np.full(n, inv_h2)       # współczynnik przy węźle poprzednim
        c = np.full(n, inv_h2)       # współczynnik przy węźle następnym
        d = np.full(n, -2.0 * inv_h2)   # część przekątnej niezależna od fali

        if self.floating:
            wall = -2.0 * inv_h2 - 2.0 * self.C_wall / (EPS0 * h)
            d[0] = d[-1] = wall
            c[0] = 2.0 * inv_h2
            a[-1] = 2.0 * inv_h2

        self._a = a
        self._c = c

        # Przekątna zależy od fali, bo każda z nich wnosi własny człon.
        self._b = d[:, None] + self.lam[None, :]

    def _factorize(self):
        """Wylicza pomocnicze współczynniki metody przemiatania.
        """
        n = self.n_int
        cp = np.empty((n, self.n_modes))
    
        inv_m = np.empty((n, self.n_modes)) 
        m = self._b[0].copy()
        
        inv_m[0] = 1.0 / m
        cp[0] = self._c[0] * inv_m[0]
        
        for i in range(1, n):
            m = self._b[i] - self._a[i] * cp[i - 1]
            inv_m[i] = 1.0 / m
            cp[i] = self._c[i] * inv_m[i]
        
        self._cp = cp
        self._inv_m = inv_m

    def _solve_tridiag(self, rhs):
        """Rozwiązuje zadanie w poprzek komory osobno dla każdej fali."""
        n = self.n_int
        
        cp, inv_m, a = self._cp, self._inv_m, self._a
        dp = np.empty_like(rhs)
        dp[0] = rhs[0] * inv_m[0]
        
        for i in range(1, n):
            dp[i] = (rhs[i] - a[i] * dp[i - 1]) * inv_m[i]
        
        x = np.empty_like(rhs)
        x[-1] = dp[-1]
        
        for i in range(n - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]
        
        return x

    # ---- rozwiązanie ----
    def solve(self, rho, phi_wall_lo=0.0, phi_wall_hi=0.0,
              sigma_lo=None, sigma_hi=None):
        """Zwraca potencjał oraz obie składowe pola dla podanego rozkładu ładunku.

        Przy ściankach o zadanym potencjale podajemy ten potencjał. przy
        ściankach pływających podajemy zamiast tego gęstość ładunku zebranego na
        każdej z nich, osobno dla każdego miejsca na obwodzie.
        """
        if self.floating:
            src = -rho / EPS0
            # Węzły na ściankach obejmują pół komórki, więc zebrany na nich
            # ladunek powierzchniowy wchodzi do bilansu z 2x waga
            
            f = 2.0 / (EPS0 * self.h_r)
            
            if sigma_lo is not None:
                src[0] = src[0] - f * sigma_lo
            
            if sigma_hi is not None:
                src[-1] = src[-1] - f * sigma_hi
            
            rhs = np.fft.rfft(src, axis=1)
            phi = np.fft.irfft(self._solve_tridiag(rhs), n=self.N_th, axis=1)
        else:
            src = -rho[1:self.N_r] / EPS0
            rhs = np.fft.rfft(src, axis=1)
            
            # Wkład znanego potencjału obu ścianek. Ścianka ma ten sam potencjał
            # na całym obwodzie, więc dokłada tylko do fali zerowej.
            rhs[0, 0] -= float(phi_wall_lo) * self.N_th / self.h_r**2
            rhs[-1, 0] -= float(phi_wall_hi) * self.N_th / self.h_r**2
            
            phi_int = np.fft.irfft(self._solve_tridiag(rhs), n=self.N_th, axis=1)

            phi = np.empty((self.N_r + 1, self.N_th))
            phi[0, :] = phi_wall_lo
            
            phi[-1, :] = phi_wall_hi
            phi[1:self.N_r, :] = phi_int

        E_r, E_th = self.gradient(phi)
        return phi, E_r, E_th

    def gradient(self, phi):
        """Liczy pole jako nachylenie potencjału ze znakiem minus.

        Wokół komory sąsiada zza brzegu bierzemy z drugiej strony, bo kierunek
        zawija się sam w sobie. W poprzek komory na samych ściankach korzystamy z
        różnicy jednostronnej.
        """
        E_r = np.empty_like(phi)
        
        E_r[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) / (2.0 * self.h_r)
        E_r[0, :] = -(phi[1, :] - phi[0, :]) / self.h_r
        E_r[-1, :] = -(phi[-1, :] - phi[-2, :]) / self.h_r

        E_th = -(np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * self.h_th)
        return E_r, E_th

    # ---- rozłożenie pola na składniki ----
    @staticmethod
    def decompose(E_r, E_th):
        """Rozkłada pole na część uśrednioną wokół komory i na zaburzenie.

        Część uśredniona zależy tylko od odległości od ścianki i opisuje warstwę
        przyścienną, czyli mechanizm ścianek. Zaburzenie to wszystko, co od tej
        średniej odstaje, czyli mechanizm turbulentny. W kierunku wokół komory
        średnia jest z definicji zerowa, bo potencjał jest tam okresowy, więc
        całe pole wokół komory jest zaburzeniem.

        Przy plywajacej sciance case jest ttudniejszy do odczytania ładunek zebrany na ściance
        nie rozkłada się równo po obwodzie, więc sama ścianka wnosi teraz coś do
        części nazywanej zaburzeniem. Miarą tego wkładu jest zmienność
        potencjału ścianki wzdłuż obwodu i dlatego zebrac trzeba ją osobno w
        diagnostyce.

        Zwraca uśrednioną składową w poprzek jako mapę o tym samym kształcie co
        pole, żeby dało się jej używać dokładnie tak samo jak pola pełnego.
        """
        E_r_mean_1d = E_r.mean(axis=1)
        E_r_mean = np.repeat(E_r_mean_1d[:, None], E_r.shape[1], axis=1)
        
        E_r_fluc = E_r - E_r_mean
        E_th_mean_1d = E_th.mean(axis=1)          # z założenia zero, kontrolka :)
        
        E_th_fluc = E_th - E_th_mean_1d[:, None]
        return E_r_mean, E_r_fluc, E_th_fluc, E_th_mean_1d

    def field_energy(self, E_r, E_th):
        """Energia zgromadzona w polu elektrycznym w płaszczyźnie, na jednostkę długości osi."""
        # Węzły na ściankach obejmują pół komórki, więc liczą się z połową wagi.
        wgt = np.ones(E_r.shape[0])
        wgt[0] = wgt[-1] = 0.5
        dens = 0.5 * EPS0 * (E_r**2 + E_th**2)
        
        return float(np.sum(dens * wgt[:, None]) * self.h_r * self.h_th)
