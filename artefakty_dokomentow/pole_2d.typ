#set document(title: "Pole elektryczne 2D – równania")
#set page(paper: "a4", margin: 2cm, numbering: "1")
#set text(lang: "pl", size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")

#let src(s) = text(size: 9pt, fill: luma(90))[(#raw(s))]

#align(center)[
  #text(16pt, weight: "bold")[Pole elektryczne 2D w `hall_pic_rt`] \
  ]

= Geometria i oznaczenia

Płaszczyzna obliczeń to $(r, theta)$:
- $r in [0, L_r]$ - w poprzek kanału, między dwiema ściankami (w kodzie `x1`),
- $theta in [0, L_theta)$ - wokół kanału, okresowo; $theta$ jest tu współrzędną długości łuku w metrach (w kodzie `x2`),
- $z$ - wzdłuż osi kanału, poza siatką (w kodzie `x3`).

Krzywizna kanału jest pominięta (geometria kartezjańska), bo $L_r << R$.

Siatka:
$ h_r = L_r / N_r, quad h_theta = L_theta / N_theta, quad i = 0, dots, N_r, quad j = 0, dots, N_theta - 1. $

Pola:
- $E_r, E_theta$ - samouzgodnione, z równania Poissona,
- $E_z$ - zadane z zewnątrz (stała `cfg.E_z` albo z modelu obwodu), jednorodne,
- $bold(B) = B_r hat(bold(r))$ - zewnętrzne.

Dryf $bold(E) times bold(B)$ idzie wzdłuż $theta$:
$ v_d = E_z / B_r. $

= Rozkład ładunku na siatkę (CIC)

Cząstka w położeniu $(r_p, theta_p)$ leży w komórce $(i, j)$:
$ i = floor(r_p / h_r), quad a = r_p / h_r - i, quad j = floor(theta_p / h_theta) mod N_theta, quad b = theta_p / h_theta - floor(theta_p / h_theta). $

Wagi czterech najbliższych węzłów:
$ S_(0 0) = (1-a)(1-b), quad S_(1 0) = a(1-b), quad S_(0 1) = (1-a) b, quad S_(1 1) = a b. $

Gęstość ładunku ($q$ - ładunek, $W_p$ - waga makrocząstki):
$ rho_(i,j) = 1/(h_r h_theta) sum_p q W_p S_(i j)(p). $

Węzły na ściankach ($i = 0$ i $i = N_r$) obejmują pół komórki, więc $rho$ jest w nich mnożone przez 2. #src("simulation.py:295")

= Równanie Poissona

$ (partial^2 phi)/(partial r^2) + (partial^2 phi)/(partial theta^2) = - rho / epsilon_0 $

Korzystam z dyskretyzacji 5-punktowej:
$ (phi_(i-1,j) - 2 phi_(i,j) + phi_(i+1,j))/h_r^2 + (phi_(i,j-1) - 2 phi_(i,j) + phi_(i,j+1))/h_theta^2 = - rho_(i,j) / epsilon_0 , $
z okresowością w $theta$: $phi_(i,-1) = phi_(i,N_theta - 1)$, $phi_(i,N_theta) = phi_(i,0)$.

= Rozkład na mody Fouriera w $theta$

$ phi_(i,j) = 1/N_theta sum_(k=0)^(N_theta - 1) hat(phi)_(i,k) e^(2 pi upright(i) j k slash N_theta) $

Dla modu $k$ część $theta$ wzoru pięciopunktowego staje się wartością własną: #src("pliczek poisson.py w 56")
$ lambda_k = (2 cos(2 pi k slash N_theta) - 2) / h_theta^2 <= 0 . $

Dla każdego $k$ zostaje osobne równanie 1D w $r$:
$ (hat(phi)_(i-1,k) - 2 hat(phi)_(i,k) + hat(phi)_(i+1,k))/h_r^2 + lambda_k hat(phi)_(i,k) = - hat(rho)_(i,k) / epsilon_0 . $ <eq-mode>

- $rho$ jest rzeczywiste, więc $hat(phi)_(i,N_theta - k) = hat(phi)_(i,k)^*$ i wystarczą mody $k = 0, dots, N_theta slash 2$ (`numpy.fft.rfft`).
- $k = 0$: $lambda_0 = 0$, a $hat(phi)_(i,0) slash N_theta$ to średnia potencjału po $theta$. To zwykły Poisson 1D dla średniej, liczony tym samym solverem co pozostałe mody.

== Algorytm Thomasa #src("plik poisson.py linijki 101-138")

@eq-mode to układ trójprzekątniowy z $n$ niewiadomymi $x_i = hat(phi)_(i,k)$ bo szukamy amplitudy k modu w i węźle:

$ a_i x_(i-1) + b_i x_i + c_i x_(i+1) = f_i, quad b_i = -2/h_r^2 + lambda_k quad "(wewnątrz obszaru)". $

Bo niewiadoma $x_i$ wiąże się z elementem $x_(i-1)$ na dolnej podprzekątnej $a_1$, z $x_i$ na głównej przekątnej $b_i$ oraz z $x_(i+1)$ na górnej podprzekątnej $c_i$. $h_r$ to krok w kierunku przestrzennym.  

Przemiatanie w przód:
$ m_0 = b_0, quad m_i = b_i - a_i c'_(i-1), quad c'_i = c_i / m_i, quad f'_0 = f_0 / m_0, quad f'_i = (f_i - a_i f'_(i-1)) / m_i . $

Przemiatanie wstecz:
$ x_(n-1) = f'_(n-1), quad x_i = f'_i - c'_i x_(i+1) . $

Macierz nie zmienia się w czasie, więc $m_i$ i $c'_i$ liczone są raz, na starcie. W każdym kroku zmienia się tylko prawa strona $f_i$. Pętla idzie po $i$, a każda iteracja liczy wszystkie mody $k$ naraz.

= Warunki na ściankach

== Case ścianki o zadanym potencjale (`czyli wall_bc = "fixed"`)

$ phi_(0,j) = phi_"lo", quad phi_(N_r,j) = phi_"hi" $

Niewiadome to $i = 1, dots, N_r - 1$. Znany potencjał przechodzi na prawą stronę, np. dla $i = 1$:
$ (-2 hat(phi)_(1,k) + hat(phi)_(2,k))/h_r^2 + lambda_k hat(phi)_(1,k) = - hat(rho)_(1,k) / epsilon_0 - hat(phi)_("lo",k) / h_r^2 . $

Potencjał ścianki jest stały wzdłuż $theta$, więc jego transformata jest niezerowa tylko dla $k = 0$ (`rfft` nie normalizuje): #src("poisson.py:170")
$ hat(phi)_("lo",k) = N_theta phi_"lo" delta_(k 0) . $

== Ścianka dielektryczna (`wall_bc = "floating"`)

Warstwa ceramiki o grubości $d$ i przenikalności $epsilon_r$ leży na uziemionym korpusie. Jej pojemność na jednostkę powierzchni: #src("poisson.py:48")
$ C = (epsilon_0 epsilon_r) / d . $

Na powierzchni gromadzi się ładunek $sigma(theta)$ zebrany z plazmy. Prawo Gaussa na powierzchni ścianki daje warunek typu Robina:
$ - epsilon_0 partial_r phi(0, theta) + C phi(0, theta) &= sigma_"lo" (theta), \
  epsilon_0 partial_r phi(L_r, theta) + C phi(L_r, theta) &= sigma_"hi" (theta). $

Całkując równanie Poissona po pół-komórce $[0, h_r slash 2]$ dostajemy wiersz dla węzła ścianki: #src("poisson.py:89-93, 150-160")
$ (2(hat(phi)_(1,k) - hat(phi)_(0,k)))/h_r^2 - (2 C)/(epsilon_0 h_r) hat(phi)_(0,k) + lambda_k hat(phi)_(0,k) = - hat(rho)_(0,k) / epsilon_0 - 2/(epsilon_0 h_r) hat(sigma)_("lo",k) $
i analogicznie dla $i = N_r$. Tu $rho_(0,j)$ to gęstość w pół-komórce, czyli wartość pomnożona przez 2 z punktu 2.

- Niewiadome to wszystkie węzły $i = 0, dots, N_r$. Potencjał ścianki jest wynikiem i zmienia się wzdłuż $theta$.
- $C$ jest takie samo na całym obwodzie, więc warunek nie miesza modów i FFT nadal działa.
- Granica $C -> oo$ daje ściankę uziemioną, $phi_w = 0$.

Ładunek cząstki o wadze $W_p$, która trafia w ściankę, jest rozkładany na dwa najbliższe węzły $theta$: #src("walls.py:39")
$ sigma_j <- sigma_j + (q W_p (1 - b)) / h_theta, quad sigma_(j+1) <- sigma_(j+1) + (q W_p b) / h_theta . $

= Pole z potencjału #src("poisson.py:184")

$ E_(r,i,j) = - (phi_(i+1,j) - phi_(i-1,j))/(2 h_r), quad i = 1, dots, N_r - 1, $
$ E_(r,0,j) = - (phi_(1,j) - phi_(0,j))/h_r, quad E_(r,N_r,j) = - (phi_(N_r,j) - phi_(N_r - 1,j))/h_r, $
$ E_(theta,i,j) = - (phi_(i,j+1) - phi_(i,j-1))/(2 h_theta) quad "(okresowo)". $

Suma po okresie się teleskopuje, więc $sum_j E_(theta,i,j) = 0$ dokładnie.

= Rozkład pola na składowe #src("poisson.py:202")

$ overline(E)_r (r_i) = 1/N_theta sum_j E_(r,i,j), quad delta E_r = E_r - overline(E)_r, quad delta E_theta = E_theta . $

- $overline(E)_r$ - warstwa przyścienna,
- $delta E_r, delta E_theta$ - zaburzenia (turbulencja).

Przy `field_mode` innym niż `"full"` cząstki czują tylko $overline(E)_r$ i $E_z$.

= Odczyt pola i ruch cząstek #src("kernels.py:69")

Pole w miejscu cząstki liczone jest tymi samymi wagami podziału na węzły siatki $S$ co przy rozkładzie ładunku:
$ E(r_p, theta_p) = sum_"4 węzły" S dot E_"węzeł" . $

Cząstka czuje pole
$ bold(E) = (overline(E)_r + delta E_r, quad delta E_theta, quad E_z) . $

Potem krok Borisa:
$ bold(v)^(-) &= bold(v)^n + (q Delta t)/(2m) bold(E), \
  bold(t) &= (q Delta t)/(2m) bold(B), quad bold(s) = (2 bold(t))/(1 + |bold(t)|^2), \
  bold(v)' &= bold(v)^(-) + bold(v)^(-) times bold(t), quad bold(v)^(+) = bold(v)^(-) + bold(v)' times bold(s), \
  bold(v)^(n+1) &= bold(v)^(+) + (q Delta t)/(2m) bold(E), quad bold(x)^(n+1) = bold(x)^n + bold(v)^(n+1) Delta t . $

= Praca poszczególnych składowych pola #src("kernels.py:173-180")

Obrót nie zmienia długości prędkości, $|bold(v)^(+)| = |bold(v)^(-)|$, więc zmiana energii w kroku wynosi
$ Delta K = (q Delta t)/2 (bold(v)^n + bold(v)^(+)) dot bold(E) + (q^2 Delta t^2)/(4m) |bold(E)|^2 = bold(k) dot bold(E), $
$ bold(k) = (q Delta t)/2 (bold(v)^n + bold(v)^(+)) + (q^2 Delta t^2)/(4m) bold(E) . $

Rozdział na składowe:
$ W_"sheath" = overline(E)_r k_r, quad W_("turb",r) = delta E_r k_r, quad W_("turb",theta) = delta E_theta k_theta, quad W_"axial" = E_z k_z, $
$ W_"sheath" + W_("turb",r) + W_("turb",theta) + W_"axial" = Delta K quad $

= Energia pola #src("poisson.py:229")

Na jednostkę długości wzdłuż $z$:
$ U = epsilon_0 / 2 sum_(i,j) g_i (E_(r,i,j)^2 + E_(theta,i,j)^2) h_r h_theta, quad g_0 = g_(N_r) = 1/2, quad g_i = 1 "dla pozostałych" i . $

!!!!!! w trybie `floating` nie jest tu wliczona energia zgromadzona na ceramice,
$ U_"cer" = sum_j C/2 (phi_(0,j)^2 + phi_(N_r,j)^2) h_theta . $
