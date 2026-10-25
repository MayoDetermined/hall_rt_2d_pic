"""Nastawy symulacji na płaszczyźnie w poprzek i wokół komory.

Płaszczyzna obliczeń obejmuje kierunek w poprzek komory, od jednej ścianki do
drugiej, oraz kierunek wokół komory, który zawija się sam w sobie. Oś kanału
zostaje poza płaszczyzną: nie liczymy wzdłuż niej położenia, ale trzymamy
składową prędkości w tym kierunku i przykładamy wzdłuż niego zadane z góry pole
elektryczne. Pole magnetyczne leży w płaszczyźnie, wzdłuż kierunku w poprzek.

Taki układ jest jedynym, w którym oba badane mechanizmy powstawania szybkich
elektronów występują naraz i dają się calkiem senownie rozdzielić. Pole osiowe razem z polem
magnetycznym daje elektronom szybki dryf wokół komory, a ten dryf rozkręca rozwazane
niestabilność. Jednocześnie obie ścianki leżą w płaszczyźnie, więc widać przy
nich warstwy przyścienne i emisję wtórną. Kierunek dryfu wychodzi z iloczynu
wektorowego pola osiowego i promieniowego, więc biegnie dokładnie wokół komory,
a zaburzenie pola wokół komory przerzuca elektrony w poprzek, na ścianki.
"""

from dataclasses import dataclass, field
import numpy as np

from hall_pic_rt.constants import E_CHARGE, M_ELECTRON, M_XENON, EPS0


@dataclass
class ConfigRT:
    """Pełna konfiguracja przebiegu na płaszczyźnie w poprzek i wokół komory."""

    # ---- Geometria ----
    L_r: float = 6.4e-3          # odległość między ściankami
    L_th: float = 6.4e-3         # jak szeroki wycinek wokół komory liczymy
    R_mean: float = 0.0425       # średni promień komory, do przeliczeń na pełny obwód
    depth_z: float = 0.01        # umowna długość wzdłuż osi, mnożnik wielkości bezwzględnych

    N_r: int = 64                # komórki w poprzek
    N_th: int = 64               # komórki wokół

    # ---- Pola narzucone z zewnątrz ----
    B_r: float = 0.02            # pole magnetyczne w poprzek komory, leży w płaszczyźnie
    E_z: float = 1.0e4           # pole elektryczne wzdłuż osi, prostopadłe do płaszczyzny

    # Ucieczka wzdłuż osi komory. Płaszczyzna obliczeń nie rozwiązuje położenia
    # wzdłuż osi, więc bez tego członu czastka nie ma jak opuścić obszaru inaczej
    # niż przez ściankę. W realnym silniku gorace elektrony wypływają wzdłuż
    # osi i to chyba wlasnie ogranicza grzanie. bez tej drogi ucieczki temperatura
    # rośnie bardziej, niż powinna.
    #
    # Zamiast usuwać czastki ze stałym prawdopodobieństwem, dajemy każdej z nich
    # pomocnicze położenie wzdłuż osi, przesuwane jej własną prędkością w tym
    # kierunku. Czastka ginie, gdy wyjdzie poza zadaną długość. Różnica jest
    # istotna w tym sensie ze strata staje się wtedy zależna od prędkości i od transportu, 
    # więc otyczy przede wszystkim tych czastek, które naprawdę mają uciekać, a nie
    # wszystkich po równo. Zero oczywiscie wyłącza ten czlon.
    axial_length: float = 0.0

    # ---- Plazma ----
    n0: float = 5.0e16           # gęstość początkowa
    Te0_eV: float = 10.0         # temperatura elektronów na starcie
    Ti0_eV: float = 0.5          # temperatura jonów na starcie
    m_ion: float = M_XENON
    n_ppc: int = 40              # czastki modelowe na komórkę i gatunek

    # ---- Podtrzymanie plazmy ----
    # Ścianki pochłaniają czastki, więc bez dosyłu plazma by zgasła. Dosyłamy
    # pary elektron-jon ze stałą wydajnością, rozłożoną równomiernie. Wartość
    # domyślna bierze się z równowagi z ubytkiem na ściankach liczonym prędkością
    # Bohma, więc gęstość powinna trzymać się blisko wartości początkowej.
    source_on: bool = True
    source_rate: float = 0.0     # zero oznacza wyliczenie z równowagi w __post_init__
    source_Te_eV: float = 10.0   # temperatura dosyłanych elektronów
    source_Ti_eV: float = 0.5    # temperatura dosyłanych jonów
    # "fixed"      - dosył o stałej wydajności; gęstość ustala się sama, ale
    #                dochodzi do swojej wartości powoli, bo napełnianie obszaru
    #                trwa kilka przelotów jonu przez kanał
    # "controlled" - dosył odtwarza to, co w danej chwili zabrały ścianki, i
    #                dodatkowo powoli koryguje odchyłkę od zadanej gęstości.
    #                Dzięki temu gęstość stoi od pierwszej chwili, więc przebieg
    #                nie marnuje czasu na napełnianie, a co ważniejsze wszystkie
    #                porównywane przebiegi mają tę samą gęstość. Bez tego
    #                gęstość różni się między nimi i staje się zmienną uboczną,
    #                której wpływu nie da się oddzielić od badanego.
    source_mode: str = "fixed"
    source_target_n: float = 0.0     # zero oznacza gęstość początkową
    source_relax_time: float = 2.0e-7   # jak szybko korygujemy odchyłkę gęstości
    source_loss_window: float = 2.0e-8  # po jakim czasie uśredniamy ubytek

    # Skąd bierze się nowa plazma.
    # "pairs"       - dosył gotowych par elektron-jon, jak wyżej.
    # "ionization"  - plazma powstaje wyłącznie z jonizacji ksenonu przez
    #                 elektrony, czyli tak jak w silniku. Gęstość gazu dobiera
    #                 się wtedy tak, by jonizacja równoważyła straty, a brakujące
    #                 elektrony wpuszcza katoda, dokładnie tak jak w silniku.
    plasma_source: str = "pairs"

    # Gęstość gazu roboczego przy pełnej jonizacji.
    # "fixed"      - gęstość zadana z góry i stała.
    # "controlled" - gęstość gazu jest powoli dostrajana tak, by gęstość plazmy
    #                trzymała się zadanej. Jest to odpowiednik pokrętła od
    #                przepływu propelantu i pełni tę samą rolę, co sterowany
    #                dosył par: sprawia, że wszystkie porównywane przebiegi mają
    #                tę samą gęstość plazmy, więc gęstość nie staje się
    #                niekontrolowaną zmienną uboczną badania.
    neutral_mode: str = "controlled"
    neutral_relax_time: float = 5.0e-7
    neutral_min: float = 1.0e17
    neutral_max: float = 1.0e22

    # Katoda. Jonizacja tworzy pary, więc sama z siebie nie łamie obojętności
    # plazmy, ale elektrony uciekają wzdłuż osi znacznie szybciej niż ciężkie
    # jony. Tę różnicę pokrywa w prawdziwym silniku prąd z katody i to samo
    # robimy tutaj: wpuszczamy tyle elektronów, ile trzeba, by ładunek w
    # obszarze wraz z ładunkiem zebranym na ściankach schodził do zera.
    cathode_on: bool = True
    cathode_Te_eV: float = 3.0
    cathode_relax_time: float = 2.0e-8

    # ---- Zewnętrzny obwód ----
    # "off"     - pole wzdłuż osi zadane z góry wartością E_z.
    # "voltage" - zasilacz o zadanym napięciu, podany przez opór i cewkę, z
    #             kondensatorem równolegle do wyładowania.
    # "current" - zasilacz o zadanym prądzie, sam dobierający napięcie.
    circuit_mode: str = "off"

    # Napięcie zasilacza 
    V_ps: float = 100.0
    # Wartości elementów podane są w skali modelu (trzeba dobrac przelicznik), a nie prawdziwego
    # silnika. Liczony wycinek ma kanał węższy i plazmę rzadszą, więc prąd
    # wychodzi około stu razy mniejszy niż cztery ampery prawdziwego wyładowania.
    # Żeby obwód zachowywał się względem tego prądu tak samo jak prawdziwy
    # względem swojego, opór i cewkę zwiększamy w tym samym stosunku, a
    # pojemność zmniejszamy. Wartości poniżej odpowiadają typowemu układowi
    # zasilania, czyli oporowi rzędu oma, cewce rzędu setek mikrohenrów i
    # pojemności rzędu mikrofarada, przeliczonym na prąd tego modelu.
    R_circ: float = 150.0
    L_circ: float = 1.0e-2
    C_circ: float = 5.0e-9

    # Zadany prąd wyładowania. Zero albo wartość ujemna oznacza, że zamiast
    # narzucać wartość z góry, obwód po czasie ustalania sam przyjmie za zadany
    # ten prąd, który akurat płynie. Jest to potrzebne, bo liczony wycinek ma
    # kanał węższy i plazmę rzadszą niż prawdziwy silnik, więc przeliczony na
    # cały pierścień prąd wychodzi o dwa rzędy wielkości mniejszy niż cztery
    # ampery prawdziwego wyładowania. wpisanie tam czterech amperów kazałoby
    # obwodowi gonić wartość, której ten model nie jest w stanie osiągnąć.
    I_set: float = 0.0
    circuit_settle_time: float = 2.0e-7
    I_L_init: float = 0.0
    V_C_init: float = -1.0           # wartość ujemna oznacza start od napięcia zasilacza
    circuit_substeps: int = 4

    # Po jakim czasie uśredniamy prąd wyładowania. Liczony wycinek obejmuje
    # drobny ułamek obwodu kanału, więc czastki odchodzą z niego pojedynczo i
    # prąd liczony z jednego kroku jest niemal samym szumem. Prawdziwy obwód
    # widzi cały pierścień naraz, czyli sumę bardzo wielu takich wycinków, i to
    # uśrednienie ją odtwarza. Okno musi być krótsze od czasu, w jakim zmienia
    # się samo wyładowanie, a dłuższe od odstępu między odejściami czastek.
    circuit_current_window: float = 2.0e-8
    V_max: float = 1.0e3             # zabezpieczenie przed rozbieganiem się obwodu

    # ---- Szybkość liczenia ----
    # Jony są ogromnie razy cięższe od elektronów, więc w ciągu jednego
    # kroku prawie się nie ruszają i kusi, żeby pchać je rzadziej. Wolno tak
    # jednak tylko wtedy, gdy jony nadążają za falą, a nie tylko za własnym
    # ruchem. Zaburzenie jest niesione przez dryf elektronów wokół komory, więc
    # wzór fali przemieszcza się o jedną komórkę w czasie równym szerokości
    # komórki podzielonej przez prędkość dryfu. Krok jonowy musi być od tego
    # czasu wyraźnie krótszy zeby się wszystko zgadzało.
    #
    # Przy settingsach badania krok jonowy okolo dziesięciokrotny łamie ten warunek i
    # widać to w wyniku, ze temperatura tła wychodzi wtedy o blisko połowę
    # niższa, a pole zaburzeń o połowę słabsze. Dlatego wartość jest sprawdzana.
    # jeden oznacza pchanie jonów co krok.
    ion_subcycle: int = 1

    # Jaki ułamek czasu przejścia fali przez komórkę wolno zająć krokowi jonów.
    ion_subcycle_limit: float = 0.2

    # Sposób upychania tablic po usunięciu czastek. Stateczny zachowuje
    # kolejność i pozwala odtworzyć dawne runy.
    stable_compact: bool = False

    # ---- Ścianki ----
    # "absorb"   - czastka ginie na ściance, bez emisji wtórnej
    # "see"      - czastka ginie, ale wybija elektrony wtórne
    # "specular" - czastka odbija się sprężyście, ścianka nic nie traci ani nie daje
    wall_mode: str = "see"
    wall_phi: float = 0.0        # potencjał obu ścianek, gdy jest zadany z góry

    # Skąd bierze się potencjał ścianki.
    # "fixed"    - ścianka utrzymywana na zadanej wartości. 
    # "floating" - ceramika o zadanej grubości leży na uziemionym korpusie i
    #              zbiera ładunek z plazmy. Jej potencjał wynika z tego, ile
    #              ładunku zebrała, i zmienia się wzdłuż obwodu komory.
    wall_bc: str = "fixed"
    wall_eps_r: float = 4.0      # przenikalność ceramiki
    wall_thickness: float = 2.0e-3   # grubość warstwy ceramicznej
    # Wydajność emisji wtórnej dla ceramiki , rosnąca liniowo z energią,
    # z nasyceniem wynikającym z ograniczenia ładunkiem przestrzennym.
    see_gamma0: float = 0.578
    see_slope: float = 0.00534   # przyrost na eV
    see_max: float = 0.986
    see_T_eV: float = 2.0        # temperatura elektronów wtórnych

    # ---- Pole samouzgodnione ----
    # "full"      - czastki widzą pełne pole z rozkładu ładunku
    # "mean_only" - czastki widzą tylko część uśrednioną wokół komory, czyli
    #               warstwę przyścienną bez zaburzeń. Tym wyłączamy turbulencję,
    #               zostawiając całą resztę bez zmian.
    field_mode: str = "full"

    # ---- Zderzenia ----
    enable_coulomb_ee: bool = True    # zderzenia kulombowskie elektron-elektron
    # Zderzenia kulombowskie rozgrywamy rzadziej niż co krok, za to od razu za
    # kilkanaście kroków naraz. Robie tak, bo czas zderzen jest krotszy.
    coulomb_interval: int = 20
    coulomb_lnL_min: float = 2.0
    coulomb_lnL_max: float = 25.0

    enable_mcc: bool = False          # zderzenia z zamrożonym gazem obojętnym
    n_neutral: float = 1.0e19         # gęstość gazu, gdy zderzenia z gazem są włączone
    T_neutral: float = 500.0
    eps_max_grid_eV: float = 1000.0
    n_energy_grid: int = 2000

    # ---- Metryczka pochodzenia i rachunek energii ----
    track_origin: bool = True         # prowadzenie osobnego rachunku dla każdej czastki
    # Próg, od którego elektron liczy się jako szybki. Musi leżeć wyraźnie
    # powyżej temperatury tła, inaczej „nadwyżka” mierzy głównie sam rozkład
    # cieplny, a nie ogon. Przy tle rzędu dziesięciu elektronowoltów rozsądny
    # jest próg czterech, pięciu jego krotności.
    E_fast_eV: float = 50.0
    eedf_bulk_eV: float = 25.0        # do tej energii dopasowujemy tło do rozkładu cieplnego
    # Nadwyżkę mierzymy przy kilku progach naraz, bo wniosek nie powinien
    # zależeć od jednego wybranego progu.
    E_fast_list: tuple = (30.0, 50.0, 80.0)
    # Progi wyrażone też jako krotność zmierzonej temperatury tła, żeby
    # porównywać przebiegi o różnej temperaturze.
    fast_over_Te: tuple = (3.0, 5.0)

    # ---- Krok i długość przebiegu ----
    dt: float = 5.0e-12
    t_end: float = 2.0e-6

    # ---- Zapis i podgląd ----
    live_view: bool = False
    plot_interval: int = 2000
    headless_save_dir: str = ""
    diag_interval: int = 200          # co ile kroków zbieramy szereg czasowy
    snapshot_times: tuple = ()        # chwile, w których zapisujemy pełny stan
    max_scatter_points: int = 6000

    seed: int = 12345
    label: str = "run"

    def __post_init__(self):
        """Dolicza wielkości pochodne i sprawdza spójność nastaw."""
        if self.wall_mode not in ("absorb", "see", "specular"):
            raise ValueError("wall_mode musi być 'absorb', 'see' albo 'specular'")
        
        if self.source_mode not in ("fixed", "controlled"):
            raise ValueError("source_mode musi być 'fixed' albo 'controlled'")
        
        if self.field_mode not in ("full", "mean_only"):
            raise ValueError("field_mode musi być 'full' albo 'mean_only'")
        
        if self.wall_bc not in ("fixed", "floating"):
            raise ValueError("wall_bc musi być 'fixed' albo 'floating'")
        
        if self.plasma_source not in ("pairs", "ionization"):
            raise ValueError("plasma_source musi być 'pairs' albo 'ionization'")
        
        if self.neutral_mode not in ("fixed", "controlled"):
            raise ValueError("neutral_mode musi być 'fixed' albo 'controlled'")
        
        if self.circuit_mode not in ("off", "voltage", "current"):
            raise ValueError("circuit_mode musi być 'off', 'voltage' albo 'current'")
        
        if self.wall_bc == "floating" and self.wall_thickness <= 0.0:
            raise ValueError("ścianka pływająca wymaga dodatniej grubości ceramiki")
        
        if self.circuit_mode != "off" and self.axial_length <= 0.0:
            raise ValueError("obwód wymaga dodatniej długości obszaru wzdłuż osi, "
                             "bo z niej bierze się napięcie i prąd wyładowania")
        if self.plasma_source == "ionization":
            # Przy pełnej jonizacji plazma bierze się wyłącznie ze zderzeń z
            # gazem, więc zderzenia z gazem muszą być włączone, a sztuczny dosył
            # par wyłączony. Ustawiamy to tutaj, żeby nie dało się przez
            # przeoczenie policzyć przebiegu z oboma źródłami naraz.
            self.enable_mcc = True
            self.source_on = False
        if self.ion_subcycle < 1:
            raise ValueError("ion_subcycle musi być dodatnią liczbą całkowitą")

        self.h_r = self.L_r / self.N_r
        self.h_th = self.L_th / self.N_th
        self.n_r_nodes = self.N_r + 1     # w poprzek są dwie ścianki, więc węzeł brzegowy po obu stronach
        self.n_th_nodes = self.N_th       # wokół komory ostatni węzeł pokrywa się z pierwszym
        self.n_cells = self.N_r * self.N_th

        self.r_nodes = np.linspace(0.0, self.L_r, self.n_r_nodes)
        self.th_nodes = np.arange(self.N_th) * self.h_th

        # Waga czastki modelowej liczona na jednostkę długości osi komory.
        self.w_ref = self.n0 * self.h_r * self.h_th / self.n_ppc
        # Mnożnik zamieniający wielkości liczone na jednostkę długości osi na
        # wielkości dla wycinka o zadanej długości.
        self.volume_factor = self.depth_z
        # Objętość liczonego obszaru, potrzebna przy dosyle i przy gęstościach.
        self.area = self.L_r * self.L_th

        self.n_steps = int(round(self.t_end / self.dt))

        # Domyślna wydajność dosyłu z równowagi z ubytkiem na ściankach.
        if self.source_rate <= 0.0:
            u_bohm = np.sqrt(E_CHARGE * self.Te0_eV / self.m_ion)
            self.source_rate = 2.0 * self.n0 * u_bohm / self.L_r
        if self.source_target_n <= 0.0:
            self.source_target_n = self.n0

        # Przelicznik z ładunku liczonego dla wycinka na prąd całego pierścienia
        # kanału. Liczony wycinek obejmuje szerokość L_th z całego obwodu, a
        # wielkości bezwzględne odnoszą się do odcinka depth_z wzdłuż osi.
        self.circuit_scale = self.depth_z * 2.0 * np.pi * self.R_mean / self.L_th

        # Wielkości kontrolne, wypisywane przed przebiegiem.
        self.lambda_D = np.sqrt(EPS0 * self.Te0_eV / (E_CHARGE * self.n0))
        self.omega_pe = np.sqrt(self.n0 * E_CHARGE**2 / (EPS0 * M_ELECTRON))
        self.omega_ce = E_CHARGE * self.B_r / M_ELECTRON
        self.omega_pi = np.sqrt(self.n0 * E_CHARGE**2 / (EPS0 * self.m_ion))
        self.v_th_e = np.sqrt(E_CHARGE * self.Te0_eV / M_ELECTRON)
        self.v_drift = self.E_z / self.B_r if self.B_r != 0.0 else np.inf
        self.rho_e = self.v_th_e / self.omega_ce if self.omega_ce != 0.0 else np.inf

        # Czy jony wolno pchać rzadziej. Miarą jest to, o jaki ułamek komórki
        # przesunie się w ciągu ich kroku wzór fali niesiony dryfem elektronów.
        self.ion_step_over_wave = (self.ion_subcycle * self.dt * self.v_drift
                                   / self.h_th) if np.isfinite(self.v_drift) else np.inf
        if self.ion_subcycle > 1 and self.ion_step_over_wave > self.ion_subcycle_limit:
            k_max = max(1, int(self.ion_subcycle_limit * self.h_th
                               / (self.dt * self.v_drift)))
            raise ValueError(
                "przy tych nastawach jony nie wolno pchać co "
                f"{self.ion_subcycle} kroków: w ciągu ich kroku wzór fali "
                f"przesuwa się o {self.ion_step_over_wave:.2f} komórki, a wolno "
                f"najwyżej o {self.ion_subcycle_limit:.2f}. Sprawdzone: przy "
                "złamaniu tego warunku temperatura tła wychodzi o blisko połowę "
                f"za niska. Wolno tu najwyżej ion_subcycle = {k_max}.")

    # ---- Wielkości zależne od położenia ----
    def B_vector(self, r):
        """Zwraca trzy składowe pola magnetycznego w podanym miejscu.

        Pole jest jednorodne i skierowane w poprzek komory, czyli leży w
        płaszczyźnie obliczeń wzdłuż pierwszej osi. Zwracamy zwykłe liczby, a
        nie tablice, bo pole nie zależy od położenia; pchacz czastek radzi sobie
        z jednym i z drugim, więc niejednorodny profil wystarczy tu wstawić.
        """
        return self.B_r, 0.0, 0.0

    def checks(self):
        """Zwraca opis warunków poprawności liczenia, do wypisania przed przebiegiem."""
        return {
            "h_r / lambda_D": self.h_r / self.lambda_D,
            "h_th / lambda_D": self.h_th / self.lambda_D,
            "omega_pe * dt": self.omega_pe * self.dt,
            "omega_ce * dt": self.omega_ce * self.dt,
            "v_drift / v_th_e": self.v_drift / self.v_th_e,
            "L_th / (2 pi rho_e)": self.L_th / (2.0 * np.pi * self.rho_e),
            "1 / omega_pi [ns]": 1e9 / self.omega_pi,
        }
