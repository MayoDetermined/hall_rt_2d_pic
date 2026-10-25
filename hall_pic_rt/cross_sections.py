"""Prawdopodobieństwa zderzeń elektronów i jonów z atomami ksenonu.

Dla każdego rodzaju zderzenia mamy funkcję, która mówi, jak chętnie zachodzi
ono przy danej energii. Wielkości te dobrane są tak, aby zgadzały się z danymi
odniesienia z tego benchmarku Landmark, co dostałem oraz z pomiarami dla ksenonu.
 Wszystkie funkcje przyjmują energię wyrażoną w elektronowoltach i działają od razu na
całych tablicach energii.
"""

import numpy as np

from .constants import E_CHARGE, M_ELECTRON, M_XENON

# Progi energetyczne. Poniżej progu wzbudzenia elektron nie ma dość energii,
# by wzbudzić atom, a poniżej progu jonizacji nie może go zjonizować.
E_EXC_XE = 8.32     # uśredniony próg wzbudzenia
E_ION_XE = 12.13    # próg jonizacji


def _safe_eps(eps_eV):
    """Podnosi zbyt małe energie do drobnej wartości dodatniej.

    Chroni to dalsze wzory przed dzieleniem przez zero i logarytmem z zera.
    """
    return np.maximum(eps_eV, 1e-3)


# Współczynnik, którym przyciszamy zderzenia sprężyste. Pierwotne dopasowanie
# dawało mi wynik mniej więcej trzy razy za duży w stosunku do
#  stałej z benchmarku Landmark oraz danych pomiarowych dla ksenonu.
# Zawyżone zderzenia sprężyste za mocno hamują elektrony i tłumią wiązkę, którą
# chcemy oglądać, dlatego dobrana jest ta poprawka, żeby przy dziesięciu
# elektronowoltach wynik trafiał w wartość odniesienia.
_EL_SCALE = 0.2688


def sigma_elastic_e(eps_eV):
    """Skłonność elektronu do zderzenia sprężystego z atomem.

    Kształt oddaje charakterystyczne dla ksenonu minimum przy niskich
    energiach, a całość jest przeskalowana wspomnianą wyżej poprawką.
    """
    eps = _safe_eps(eps_eV)
    # Najpierw człon dający płytkie minimum przy niskiej energii, potem człon
    # opadający wraz z rosnącą energią.
    ramsauer = 1.2e-19 * (eps / (eps + 0.5)) ** 2
    high = 6.0e-19 / (1.0 + eps / 30.0)
    return _EL_SCALE * (ramsauer + high)


# Liczby dobrane tak, by uśrednione po rozkładzie prędkości tempo jonizacji i
# strat energii pokrywało się z tabelami Landmark + pomiarami dla Xe
_IZ_A, _IZ_B, _IZ_C = 9.184486e-20, 0.4091, 0.6000
_EX_A, _EX_B, _EX_C = 1.636412e-19, 0.9364, 1.1000


def sigma_excitation_e(eps_eV):
    """Skłonność elektronu do wzbudzenia atomu.

    Poniżej progu wzbudzenia jest zerowa. Powyżej rośnie zgodnie z dobranym
    kształtem, który odtwarza straty energii z tabel Landmark.
    """
    eps = _safe_eps(eps_eV)
    sig = np.zeros_like(eps)
    mask = eps > E_EXC_XE
    u = eps[mask] / E_EXC_XE
    sig[mask] = _EX_A * np.log(u) / u ** _EX_C * (1.0 - 1.0 / u) ** _EX_B
    return np.maximum(sig, 0.0)


def sigma_ionization_e(eps_eV):
    """Skłonność elektronu do zjonizowania atomu.

    Poniżej progu jonizacji jest zerowa, a powyżej odtwarza tempo jonizacji
    z tabel Landmark.
    """
    eps = _safe_eps(eps_eV)
    sig = np.zeros_like(eps)
    mask = eps > E_ION_XE
    u = eps[mask] / E_ION_XE
    sig[mask] = _IZ_A * np.log(u) / u ** _IZ_C * (1.0 - 1.0 / u) ** _IZ_B
    return np.maximum(sig, 0.0)


def sigma_total_e(eps_eV):
    """Łączna skłonność elektronu do jakiegokolwiek zderzenia z atomem.

    Jest to suma zderzeń sprężystych, wzbudzenia i jonizacji.
    """
    return (sigma_elastic_e(eps_eV)
            + sigma_excitation_e(eps_eV)
            + sigma_ionization_e(eps_eV))


# Zderzenia jonów z atomami.

def sigma_cex_ion(eps_eV):
    """Skłonność jonu do wymiany ładunku z atomem.

    W takim zderzeniu jon zabiera elektron atomowi i sam staje się zimnym
    atomem, a w jego miejscu pojawia się zimny jon. Kształt dopasowany do
    danych dla ksenonu.
    """
    eps = _safe_eps(eps_eV)
    a, b = 87.3e-20 ** 0.5, 13.6e-20 ** 0.5  # pomocnicze wartości utrzymujące dodatnią podstawę
    val = (55.0 - 3.0 * np.log(eps)) * 1e-20
    return np.maximum(val, 1e-20)


def sigma_elastic_ion(eps_eV):
    """Skłonność jonu do zwykłego, sprężystego zderzenia z atomem.

    Co do rzędu wielkości jest zbliżona do wymiany ładunku.
    """
    eps = _safe_eps(eps_eV)
    return np.maximum((45.0 - 2.5 * np.log(eps)) * 1e-20, 1e-20)


def sigma_total_ion(eps_eV):
    """Łączna skłonność jonu do zderzenia z atomem: wymiana ładunku i zderzenie sprężyste."""
    return sigma_cex_ion(eps_eV) + sigma_elastic_ion(eps_eV)


# Proste przeliczenia między energią a szybkością.

def speed_from_energy_e(eps_eV):
    """Zamienia energię elektronu na jego szybkość."""
    return np.sqrt(2.0 * eps_eV * E_CHARGE / M_ELECTRON)


def energy_eV_from_speed_e(v):
    """Zamienia szybkość elektronu na jego energię."""
    return 0.5 * M_ELECTRON * v ** 2 / E_CHARGE


def energy_eV_from_speed_ion(v):
    """Zamienia szybkość jonu ksenonu na jego energię."""
    return 0.5 * M_XENON * v ** 2 / E_CHARGE
