"""Symulacja czastek w komórce na płaszczyźnie w poprzek i wokół komory.

Płaszczyzna obliczeń obejmuje kierunek od ścianki do ścianki oraz kierunek
biegnący wokół komory. Oś komory zostaje poza płaszczyzną: trzymamy wzdłuż niej
składową prędkości i przykładamy zadane pole elektryczne. Dzięki temu w jednym
przebiegu występują naraz oba mechanizmy, które chcemy porównać, czyli warstwy
przy ściankach i niestabilność dryfu wokół komory.

Ten plik udostępnia trzy rzeczy: konfigurację, samą symulację oraz zestaw
narzędzi do odczytania metryczki pochodzenia.
"""

from .config import ConfigRT
from .simulation import SimulationRT
from . import origin

__all__ = ["ConfigRT", "SimulationRT", "origin"]
