"""Zestaw stalych fizycznych, ktorych uzywamy w calym projekcie.

Wszystko podane jest w jednostkach ukladu SI. Trzymamy je w jednym miejscu,
zeby nie powtarzac tych samych liczb w kilku plikach i zeby latwo bylo je
sprawdzic.
"""

# Ladunek elementarny, w kulombach.
E_CHARGE = 1.602176634e-19
# Masa spoczynkowa elektronu, w kilogramach.
M_ELECTRON = 9.1093837015e-31
# Atomowa jednostka masy, w kilogramach.
AMU = 1.66053906660e-27
# Masa pojedynczego jonu ksenonu. Ksenon jest gazem roboczym silnika.
M_XENON = 131.293 * AMU
# Przenikalnosc elektryczna prozni, w faradach na metr.
EPS0 = 8.8541878128e-12
# Stala Boltzmanna, w dzulach na kelwin.
K_BOLTZMANN = 1.380649e-23

# Przeliczniki miedzy elektronowoltami a dzulami, przydatne przy energiach.
EV_TO_JOULE = E_CHARGE
JOULE_TO_EV = 1.0 / E_CHARGE
