"""Rachunek energii całego obszaru obliczeń.

Prowadzimy taką a la księgę, w której po jednej stronie stoi zmiana energii ruchu czastek,
a po drugiej wszystko, co mogło ją spowodować: praca każdej ze składowych pola
elektrycznego z osobna, energia wniesiona przez dosył i przez elektrony wtórne,
energia zabrana przez ścianki, wymiana z gazem oraz przetasowanie w zderzeniach
między elektronami. Reszta, czyli różnica obu stron, powinna być bliska zeru.

Rozbicie pracy pola naskładowe jest metryczką tego, ile energii dostarczył każdy mechanizm

Wszystkie wielkości podajemy w J dla wycinka o zadanej długości wzdłuż osi
komory. Wielkości liczone przez resztę programu odnoszą się do jednostki tej
długości, więc mnożymy je tutaj przez jeden wspólny przelicznik.
"""

import numpy as np

from hall_pic_rt.constants import E_CHARGE


# Składowe pracy pola. Pierwsza opisuje napęd wzdłuż osi komory, druga warstwę
# przyścienną, czyli mechanizm ścianek, dwie ostatnie zaburzenia, czyli
# mechanizm turbulentny.
FIELD_TERMS = ("W_axial", "W_sheath", "W_turb_th", "W_turb_r")

TERM_LABELS = {
    "W_axial": "praca pola wzdłuż osi",
    "W_sheath": "praca warstwy przyściennej",
    "W_turb_th": "praca zaburzeń wokół komory",
    "W_turb_r": "praca zaburzeń w poprzek",
    "E_source": "dosył plazmy",
    "E_see": "elektrony wtórne ze ścianek",
    "E_wall": "ubytek na ściankach",
    "E_mcc": "zderzenia z gazem",
    "E_ee": "zderzenia elektron-elektron",
}


class EnergyBook:
    """Zbiera wszystkie przepływy energii i sprawdza, czy księga się domyka."""

    def __init__(self, cfg):
        """Zakłada pustą księgę dla podanej konfiguracji."""
        self.cfg = cfg
        self.f = cfg.volume_factor          # przelicznik na wycinek o zadanej długości
        self.reset()

    def reset(self):
        """Zeruje wszystkie pozycje księgi."""
        self.terms = {}
        for sp in ("e", "i"):
            for t in FIELD_TERMS:
                self.terms[f"{t}_{sp}"] = 0.0

        self.terms["E_source_e"] = 0.0
        self.terms["E_source_i"] = 0.0

        self.terms["E_see"] = 0.0

        self.terms["E_wall_e"] = 0.0
        self.terms["E_wall_i"] = 0.0

        self.terms["E_axial_e"] = 0.0
        self.terms["E_axial_i"] = 0.0

        self.terms["E_mcc"] = 0.0
        self.terms["E_ee"] = 0.0

        self.KE0_e = self.KE0_i = 0.0

    def kinetic(self, sp):
        """Energia ruchu jednego gatunku, w dżulach."""
        if sp.N == 0:
            return 0.0
        return float(0.5 * sp.mass * np.sum(sp.aw * sp.speed2())) * self.f

    def set_initial(self, electrons, ions):
        """Zapamiętuje energię ruchu na starcie, od której liczymy zmianę."""
        self.KE0_e = self.kinetic(electrons)
        self.KE0_i = self.kinetic(ions)

    def add_field_work(self, species_tag, ledger_parts, weights):
        """Dopisuje pracę każdej składowej pola, zsumowaną po czastkach.

        Praca podana jest na czastkę w elektronowoltach, więc mnożymy ją przez
        wagę i zamieniamy na dżule.
        """
        for name, val in ledger_parts.items():
            self.terms[f"{name}_{species_tag}"] += float(
                np.sum(weights * val)) * E_CHARGE * self.f

    def add(self, key, value_eV):
        """Dopisuje pozycję podaną w elektronowoltach razy waga."""
        self.terms[key] += value_eV * E_CHARGE * self.f

    def field_work_total(self, species_tag=None):
        """Suma pracy wszystkich składowych pola, dla jednego lub obu gatunków."""
        tags = ("e", "i") if species_tag is None else (species_tag,)
        return sum(self.terms[f"{t}_{s}"] for t in FIELD_TERMS for s in tags)

    def summary(self, electrons, ions, U_field=0.0):
        """Zwraca pełne zestawienie księgi razem z resztą zamknięcia."""
        ke_e = self.kinetic(electrons)
        ke_i = self.kinetic(ions)
        dKE = (ke_e - self.KE0_e) + (ke_i - self.KE0_i)

        gains = (self.field_work_total()
                 + self.terms["E_source_e"] + self.terms["E_source_i"]
                 + self.terms["E_see"] + self.terms["E_mcc"] + self.terms["E_ee"])
        
        losses = (self.terms["E_wall_e"] + self.terms["E_wall_i"]
                  + self.terms["E_axial_e"] + self.terms["E_axial_i"])
        
        residual = dKE - (gains - losses)
        
        scale = max(abs(self.field_work_total()), abs(dKE), 1e-30)

        out = dict(self.terms)
        out.update({
            "KE_e": ke_e, "KE_i": ke_i, "dKE": dKE,
            "U_field": U_field * self.f,
            "gains": gains, "losses": losses,
            "residual": residual, "residual_rel": residual / scale,
        })
        
        # Udział każdego mechanizmu w pracy pola nad elektronami.
        tot_e = sum(self.terms[f"{t}_e"] for t in FIELD_TERMS)
        
        for t in FIELD_TERMS:
            out[f"frac_{t}_e"] = self.terms[f"{t}_e"] / tot_e if tot_e != 0.0 else 0.0
        return out
