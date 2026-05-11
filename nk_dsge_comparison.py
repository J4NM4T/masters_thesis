# ==============================================================================
# NK DSGE — Porownanie polityk monetarnych
#
# Skrypt laduje 3 wytrenowane modele SAC z katalogu saved_models/
# (po jednym na kazda regule Taylora) i porownuje je z 3 regulami Taylora.
#
# UWAGA METODOLOGICZNA — krytyka Lucasa:
#   Krytyka Lucasa (1976) glosi, ze gdy zmienia sie regula polityki, zmieniaja
#   sie rowniez oczekiwania sektora prywatnego. Pelne zastosowanie tej zasady
#   wymagaloby rozwiazywania PUNKTU STALEGO miedzy polityka agenta SAC a
#   oczekiwaniami racjonalnymi pod ta polityka.
#
#   W tej pracy stosujemy pragmatyczne UPROSZCZENIE (Hinterlang & Tanzer 2021,
#   Chen et al. 2023): macierz oczekiwan R_mat jest wyprowadzona pod regula
#   Taylora i pozostaje niezmieniona, gdy SAC zastepuje regule. Innymi slowy:
#   sektor prywatny "wciaz oczekuje", ze bank centralny zachowuje sie jak Taylor,
#   choc faktycznie decyduje siec neuronowa. Jest to ZNANE NARUSZENIE krytyki
#   Lucasa, akceptowane w literaturze DRL-makro dla zachowania trakowalnosci.
#   Konsekwencje tego uproszczenia omawiamy w rozdziale 4.
#
#   Ewaluujemy kazdego agenta w srodowisku, w ktorym byl trenowany (te same R_mat),
#   zeby uniknac DODATKOWEJ niespojnosci miedzy oczekiwaniami treningowymi
#   a ewaluacyjnymi.
#
# REGULY TAYLORA:
#   1. Simple (prosta):          r_t = phi_pi*pi_t + phi_y*y_t + eps_R_t
#   2. Smoothing (wygladzanie):  r_t = rho*r_{t-1} + (1-rho)*(phi_pi*pi_t + phi_y*y_t) + eps_R_t
#   3. Forward-looking:          r_t = phi_pi*E_t{pi_{t+1}} + phi_y*E_t{y_{t+1}} + eps_R_t
#
# WAZNE — macierze BK:
#   Kazda regula Taylora implikuje inne oczekiwania sektora prywatnego
#   (gospodarstw domowych i firm), poniewaz zmiana reguly zmienia caly
#   system racjonalnych oczekiwan. Dlatego macierz R_mat z dekompozycji
#   Blancharda-Kahna jest wyprowadzona osobno dla kazdej reguly w dsge.R.
#
#   - Simple i Forward: stan BK = [eps_D, eps_S, eps_R] (3-wymiarowy)
#     R_mat: 3x3 (wiersze: [pi, r, y], kolumny: [eps_D, eps_S, eps_R])
#
#   - Smoothing: stan BK = [eps_D, eps_S, eps_R, r_{t-1}] (4-wymiarowy)
#     r_{t-1} staje sie zmienna stanu (predetermined), bo regula zalezy od r[-1].
#     R_mat: 2x4 (wiersze: [pi, y], kolumny: [eps_D, eps_S, eps_R, r_{t-1}])
#     W gEcon r jest zmienną stanu, nie kontrolną — zmniejsza to wymiar R o 1 wiersz.
#
# RL AGENT:
#   3 agenty SAC (Soft Actor-Critic) — po jednym na regule Taylora.
#   Kazdy trenowany z oczekiwaniami sektora prywatnego z dopasowanej reguly:
#     SAC_simple    -> R_MAT_SIMPLE
#     SAC_smoothing -> R_MAT_SMOOTH (rozszerzony stan z r_{t-1})
#     SAC_forward   -> R_MAT_FORWARD
#   Por. Hinterlang & Tanzer (2021) dla podejscia RL-macro.
#
# WYNIKI (katalog saved_models/):
#   Kazdy wykres porownuje Taylor vs SAC W OBREBIE tego samego srodowiska.
#   Nie mieszamy srodowisk — rozne macierze BK = rozne dynamiki.
#
#   - fig1_irf_demand.png       — IRF na szok popytowy (3 panele: simple/smooth/fwd)
#   - fig2_irf_supply.png       — IRF na szok podazowy (3 panele)
#   - fig3_mc_boxplot.png       — rozklad welfare loss (MC, 3 pary Taylor vs SAC)
#   - fig4_welfare_bar.png      — slupki grupowane Taylor vs SAC + % redukcja
#   - fig5_variance_frontier.png — wariancja pi vs y (3 panele, strzalka T->SAC)
#   - fig6_timeseries.png       — szeregi czasowe (3 panele, wspolne szoki)
#   - fig7_mc_distributions.png — gestosci pi_t i y_t (3 panele)
#   - fig8_rate_smoothing.png   — Std(delta_r) slupki grupowane
#
# Referencje:
#   Gali (2015) Monetary Policy, Inflation, and the Business Cycle — rozdz. 3
#   Woodford (2003) Interest and Prices — funkcja straty
#   Poutineau et al. (2015) — kalibracja NK3
#   Hinterlang & Tanzer (2021) — RL for optimal monetary policy
# ==============================================================================

import os
import time
import multiprocessing as mp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from gymnasium import spaces


# ==============================================================================
# 0. STALE MODELU
# ==============================================================================

BETA        = 0.99
SIGMA       = 1.0
PHI         = 1.0
THETA       = 0.75
KAPPA       = ((1 - THETA) * (1 - THETA * BETA) / THETA) * (SIGMA + PHI)
SIGMA_SHOCK = 0.01
LAMBDA_Y    = KAPPA / 6.0

# Parametry regul Taylora
PHI_PI  = 1.5       # wspolczynnik reakcji na inflacje
PHI_Y   = 0.125     # wspolczynnik reakcji na luke produkcyjna
RHO_R   = 0.8       # parametr wygladzania stopy (smoothing rule)

# Mianowniki ukladu jednoczesnego (analityczne rozwiazanie):
#   r* = (phi_pi*B + phi_y*A) / D   (prosta regula)
#   r* = (rho*r_prev + (1-rho)*(phi_pi*B + phi_y*A)) / D_smooth
#   gdzie A = E_y + E_pi/sigma + eps_D,  B = beta*E_pi + kappa*A + eps_S
_D_SIMPLE = 1.0 + (PHI_PI * KAPPA + PHI_Y) / SIGMA
_D_SMOOTH = 1.0 + (1.0 - RHO_R) * (PHI_PI * KAPPA + PHI_Y) / SIGMA

# Macierz trwalosci szokow AR(1) — wspolna dla wszystkich regul
# (szoki sa egzogeniczne i nie zaleza od reguly polityki pienieznej)
P_MAT = np.diag([0.90, 0.90, 0.40])   # [rho_D, rho_S, rho_R]

SAVE_DIR   = "saved_models"
MC_WORKERS = 12     # liczba workerow Monte Carlo (= watki logiczne Ryzen 5 3600)
MC_T       = 500    # dlugosc jednej symulacji MC
MC_N_RUNS  = 200    # liczba replikacji MC


# ==============================================================================
# 1. MACIERZE BLANCHARDA-KAHNA DLA KAZDEJ REGULY TAYLORA
#
# Wyprowadzone w dsge.R (gEcon) — kazda regula osobno, bo zmiana reguly
# polityki monetarnej zmienia caly system racjonalnych oczekiwan.
# Kolejnosc wierszy/kolumn jest zgodna z gEcon: get_pert_solution().
# ==============================================================================

# --- Prosta regula Taylora ---
# r_t = phi_pi * pi_t + phi_y * y_t + eps_R_t
# Stan BK: [eps_D, eps_S, eps_R] (3-wymiarowy)
# R_mat: (3 zmienne kontrolne: pi, r, y) x (3 zmienne stanu: eps_D, eps_S, eps_R)
R_MAT_SIMPLE = np.array([
    [1.2115,  1.5879, -0.1096],   # wiersz 0: E_t{pi_{t+1}}
    [1.9134,  1.8526,  0.1875],   # wiersz 1: E_t{r_{t+1}}  (nieuzywany bezposrednio w dynamice)
    [0.7693, -4.2345, -0.3855],   # wiersz 2: E_t{y_{t+1}}
])

# --- Regula z wygladzaniem (smoothing) ---
# r_t = rho_r * r_{t-1} + (1-rho_r) * (phi_pi * pi_t + phi_y * y_t) + eps_R_t
# Stan BK: [eps_D, eps_S, eps_R, r_{t-1}] (4-wymiarowy, r jest zmienna stanu)
# R_mat: (2 zmienne kontrolne: pi, y) x (4 zmienne stanu: eps_D, eps_S, eps_R, r_{t-1})
# UWAGA: w gEcon, r staje sie zmienna stanu (predetermined) bo regula zawiera r[-1].
#         Dlatego R_mat ma 2 wiersze (pi, y), a nie 3 — r nie jest juz kontrolna.
R_MAT_SMOOTH = np.array([
    [1.4692,  1.8374, -0.6292, -0.7203],   # wiersz 0: E_t{pi_{t+1}}
    [3.0839, -1.9934, -1.4911, -1.9755],   # wiersz 1: E_t{y_{t+1}}
])

# --- Forward-looking regula Taylora ---
# r_t = phi_pi * E_t{pi_{t+1}} + phi_y * E_t{y_{t+1}} + eps_R_t
# Stan BK: [eps_D, eps_S, eps_R] (3-wymiarowy)
# R_mat: (3 zmienne kontrolne: pi, r, y) x (3 zmienne stanu: eps_D, eps_S, eps_R)
R_MAT_FORWARD = np.array([
    [1.5387,  1.9046, -0.1608],   # wiersz 0: E_t{pi_{t+1}}
    [2.1871,  2.1175,  0.2752],   # wiersz 1: E_t{r_{t+1}}  (nieuzywany bezposrednio)
    [0.9770, -4.0334, -0.5659],   # wiersz 2: E_t{y_{t+1}}
])

# ==============================================================================
# 2. KONFIGURACJA POLITYK
# ==============================================================================

# Kazda regula Taylora: macierz oczekiwan, indeksy wierszy, rozszerzony stan
POLICY_CONFIGS = {
    "Taylor\nSimple": {
        "R_mat": R_MAT_SIMPLE,
        "E_pi_row": 0,           # wiersz R_mat odpowiadajacy E_t{pi_{t+1}}
        "E_y_row": 2,            # wiersz R_mat odpowiadajacy E_t{y_{t+1}}
        "expanded_state": False,  # stan BK = [eps_D, eps_S, eps_R]
        "description": r"$r_t = 1.5\pi_t + 0.125\tilde{y}_t$",
    },
    "Taylor\nSmoothing": {
        "R_mat": R_MAT_SMOOTH,
        "E_pi_row": 0,
        "E_y_row": 1,            # w wersji smooth: wiersz 1 = y (bo r nie jest kontrolna)
        "expanded_state": True,   # stan BK = [eps_D, eps_S, eps_R, r_{t-1}]
        "description": r"$r_t = 0.8 r_{t-1} + 0.2(1.5\pi_t + 0.125\tilde{y}_t)$",
    },
    "Taylor\nForward": {
        "R_mat": R_MAT_FORWARD,
        "E_pi_row": 0,
        "E_y_row": 2,
        "expanded_state": False,
        "description": r"$r_t = 1.5 E_t\{\pi_{t+1}\} + 0.125 E_t\{\tilde{y}_{t+1}\}$",
    },
}

# Mapowanie regula -> plik modelu i VecNormalize
SAC_FILES = {
    "simple":    ("sac_simple",    "vec_normalize_simple.pkl"),
    "smoothing": ("sac_smoothing", "vec_normalize_smoothing.pkl"),
    "forward":   ("sac_forward",   "vec_normalize_forward.pkl"),
}

# Pary (srodowisko, Taylor key, SAC key) — porownania wylacznie w obrebie
# tego samego srodowiska (tych samych macierzy BK). Mieszanie srodowisk
# na jednym wykresie jest mylace, bo rozne macierze R/S generuja rozne
# dynamiki — nie mozna porownywac strat welfare miedzy srodowiskami.
PAIRS = [
    ("simple",    "Taylor\nSimple",    "SAC\nSimple"),
    ("smoothing", "Taylor\nSmoothing", "SAC\nSmoothing"),
    ("forward",   "Taylor\nForward",   "SAC\nForward"),
]
ENV_TITLES = {
    "simple":    "Srodowisko: prosta regula Taylora",
    "smoothing": "Srodowisko: regula z wygladzaniem",
    "forward":   "Srodowisko: forward-looking regula",
}

# Kolory — 2 role (Taylor vs SAC), nie 6 polityk
COLOR_TAYLOR = "#e74c3c"
COLOR_SAC    = "#27ae60"


# ==============================================================================
# 3. SRODOWISKO (potrzebne do ladowania VecNormalize)
# ==============================================================================

class NewKeynesianEnv(gym.Env):
    """
    Srodowisko-stub uzywane WYLACZNIE do rekonstrukcji VecNormalize przy
    ladowaniu modeli SAC. Faktyczna dynamika DSGE w tym pliku liczona jest
    recznie w simulate_taylor_trajectory / simulate_sac_trajectory / simulate_scenario,
    a step() ponizej nie jest wywolywany w trakcie ewaluacji.

    VecNormalize.load() wymaga wylacznie zgodnych observation_space i action_space
    oraz funkcji reset() — dlatego klasa jest celowo minimalna.
    """
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(
            low=np.array([-5.0], dtype=np.float32),
            high=np.array([5.0], dtype=np.float32),
            dtype=np.float32
        )
        high_obs = np.array([10., 10., 3., 3., 3., 5.], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high_obs, high=high_obs, dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(6, dtype=np.float32), {}

    def step(self, action):
        # Niewywolywany w ewaluacji — pelna dynamika jest w simulate_*.
        raise NotImplementedError(
            "NewKeynesianEnv.step() jest stubem; uzyj simulate_taylor_trajectory "
            "lub simulate_sac_trajectory."
        )


def _make_env():
    return NewKeynesianEnv()


# ==============================================================================
# 4. FUNKCJE REGUL TAYLORA I OCZEKIWAN
# ==============================================================================

def compute_expectations(rule_name, eps, r_current):
    """
    Oblicza oczekiwania E_t{pi_{t+1}} i E_t{y_{t+1}} z macierzy BK
    specyficznych dla danej reguly Taylora.

    Konwencja gEcon: E_t{c_{t+1}} = R @ s_t,  gdzie:
      - simple/forward: s_t = [eps_D, eps_S, eps_R]  (3-wymiarowy)
      - smoothing:      s_t = [eps_D, eps_S, eps_R, r_t]  (4-wymiarowy)

    Czwarty element dla smoothing to r_t (BIEZACA stopa procentowa),
    a nie r_{t-1} — bo r_t jest predetermined dla t+1.
    """
    cfg = POLICY_CONFIGS[rule_name]
    R_mat = cfg["R_mat"]
    if cfg["expanded_state"]:
        state_bk = np.concatenate([eps, [r_current]])
    else:
        state_bk = eps
    E_pi = float(R_mat[cfg["E_pi_row"]] @ state_bk)
    E_y  = float(R_mat[cfg["E_y_row"]]  @ state_bk)
    return E_pi, E_y


def compute_taylor_rate_and_expectations(rule_name, eps, prev_r):
    """
    Oblicza stope procentowa r_t ORAZ oczekiwania E_t{pi_{t+1}}, E_t{y_{t+1}}
    jednoczesnie (rozwiazujac uklad rownan liniowych).

    Problem jednoczesnosci dotyczy reguly SMOOTHING:
      - Oczekiwania zaleza od r_t:  E = R @ [eps, r_t]
      - r_t zalezy od oczekiwan:    r = rho*r_{t-1} + (1-rho)*(phi_pi*pi + phi_y*y)
      - pi i y zaleza od E i r (przez IS i NKPC)
    Rozwiazanie: podstawiamy linearna zaleznosc E od r do reguly Taylora
    i rozwiazujemy JEDNO rownanie liniowe na r_t.

    Dla regul simple i forward — brak jednoczesnosci (r nie jest w stanie BK).
    """
    cfg = POLICY_CONFIGS[rule_name]

    if rule_name == "Taylor\nSmoothing":
        # --- Jednoczesne rozwiazanie: E zalezy liniowo od r_t ---
        R_mat = cfg["R_mat"]
        R03 = R_mat[cfg["E_pi_row"], 3]     # dE_pi/dr_t
        R13 = R_mat[cfg["E_y_row"],  3]     # dE_y/dr_t

        # Oczekiwania w punkcie r=0 (czesc niezalezna od r_t):
        E_pi_0 = float(R_mat[cfg["E_pi_row"], :3] @ eps)
        E_y_0  = float(R_mat[cfg["E_y_row"],  :3] @ eps)

        # A(r) = E_y(r) + E_pi(r)/sigma + eps_D = A_0 + alpha_A * r
        # B(r) = beta*E_pi(r) + kappa*A(r) + eps_S = B_0 + alpha_B * r
        alpha_A = R13 + R03 / SIGMA
        alpha_B = BETA * R03 + KAPPA * alpha_A

        A_0 = E_y_0 + E_pi_0 / SIGMA + eps[0]
        B_0 = BETA * E_pi_0 + KAPPA * A_0 + eps[1]

        # r * D_smooth = rho*r_{t-1} + (1-rho)*(phi_pi*(B_0+alpha_B*r) + phi_y*(A_0+alpha_A*r))
        # r * (D_smooth - (1-rho)*(phi_pi*alpha_B + phi_y*alpha_A))
        #   = rho*r_{t-1} + (1-rho)*(phi_pi*B_0 + phi_y*A_0)
        D_adj = _D_SMOOTH - (1.0 - RHO_R) * (PHI_PI * alpha_B + PHI_Y * alpha_A)
        numer = RHO_R * prev_r + (1.0 - RHO_R) * (PHI_PI * B_0 + PHI_Y * A_0)
        r_t = float(np.clip(numer / D_adj, -5.0, 5.0))

        E_pi = E_pi_0 + R03 * r_t
        E_y  = E_y_0  + R13 * r_t
        return r_t, E_pi, E_y

    else:
        # --- Simple i Forward: brak r w stanie BK ---
        E_pi, E_y = compute_expectations(rule_name, eps, 0.0)

        if rule_name == "Taylor\nSimple":
            A = E_y + E_pi / SIGMA + eps[0]
            B = BETA * E_pi + KAPPA * A + eps[1]
            r_t = float(np.clip(
                (PHI_PI * B + PHI_Y * A) / _D_SIMPLE,
                -5.0, 5.0))
        elif rule_name == "Taylor\nForward":
            r_t = float(np.clip(
                PHI_PI * E_pi + PHI_Y * E_y,
                -5.0, 5.0))
        else:
            raise ValueError(f"Nieznana regula: {rule_name}")

        return r_t, E_pi, E_y


# ==============================================================================
# 5. RDZEN SYMULACJI
# ==============================================================================

def simulate_taylor_trajectory(rule_name, shocks, T):
    """
    Symuluje T krokow modelu NK DSGE dla reguly Taylora.
    Kazda regula uzywa wlasnych macierzy BK do formowania oczekiwan.

    Dla reguly smoothing: r_t i oczekiwania sa rozwiazywane jednoczesnie
    (r_t jest czescia stanu BK, wiec oczekiwania zaleza od r_t).

    shocks: array (T, 3) — prelosowane szoki eta_t (eta[2] = 0)
    Zwraca slownik: {"pi": [...], "y": [...], "r": [...], "loss": float}
    """
    eps = np.zeros(3)
    pi, y, prev_r = 0.0, 0.0, 0.0
    hist_pi, hist_y, hist_r = [], [], []
    cumulative_loss = 0.0

    for t in range(T):
        eps = P_MAT @ eps + shocks[t]

        r, E_pi, E_y = compute_taylor_rate_and_expectations(
            rule_name, eps, prev_r)

        y_t  = E_y - (1.0 / SIGMA) * (r - E_pi) + eps[0]
        pi_t = BETA * E_pi + KAPPA * y_t + eps[1]

        cumulative_loss += (BETA ** t) * (pi_t**2 + LAMBDA_Y * y_t**2)

        hist_pi.append(pi_t)
        hist_y.append(y_t)
        hist_r.append(r)
        pi, y, prev_r = pi_t, y_t, r

    return {"pi": hist_pi, "y": hist_y, "r": hist_r, "loss": cumulative_loss}


def simulate_sac_trajectory(model, eval_env, shocks, T, rule_name):
    """
    Symuluje T krokow modelu NK DSGE pod polityka SAC wytrenowana
    dla konkretnej reguly (rule_name identyfikuje srodowisko oczekiwan).
    """
    eps = np.zeros(3)
    pi, y, prev_r = 0.0, 0.0, 0.0
    hist_pi, hist_y, hist_r = [], [], []
    cumulative_loss = 0.0

    for t in range(T):
        eps = P_MAT @ eps + shocks[t]

        state  = np.array([y, pi, eps[0], eps[1], eps[2], prev_r], dtype=np.float32)
        norm_s = eval_env.normalize_obs(state.reshape(1, -1))
        action, _ = model.predict(norm_s, deterministic=True)
        # SB3 SAC zwraca akcje w przedziale action_space=Box(-5,5); dodatkowy clip
        # jest zbedny i tworzy plaski obszar gradientu (defensywnie zachowany jako
        # asercja).
        r = float(action[0][0])
        assert -5.0 - 1e-6 <= r <= 5.0 + 1e-6, f"SAC zwrocil akcje poza zakresem: {r}"

        E_pi, E_y = compute_expectations(rule_name, eps, r)

        y_t  = E_y - (1.0 / SIGMA) * (r - E_pi) + eps[0]
        pi_t = BETA * E_pi + KAPPA * y_t + eps[1]

        cumulative_loss += (BETA ** t) * (pi_t**2 + LAMBDA_Y * y_t**2)

        hist_pi.append(pi_t)
        hist_y.append(y_t)
        hist_r.append(r)
        pi, y, prev_r = pi_t, y_t, r

    return {"pi": hist_pi, "y": hist_y, "r": hist_r, "loss": cumulative_loss}


# ==============================================================================
# 6. WORKER MONTE CARLO — wywolywany przez mp.Pool
# ==============================================================================

# Cache modeli w worker'ze pool'a — wypelniany raz przy pierwszym wywolaniu
# danego (model_path, vec_path) i reuzywany dla pozostalych replikacji MC.
# Wczesniej kazdy z 600 wywolan workera deserializowal cala siec neuronowa.
_WORKER_CACHE = {}


def _get_or_load(model_path, vec_path):
    key = (model_path, vec_path)
    if key not in _WORKER_CACHE:
        loaded_model = SAC.load(model_path)
        eval_env = VecNormalize.load(vec_path, DummyVecEnv([_make_env]))
        eval_env.training    = False
        eval_env.norm_reward = False
        _WORKER_CACHE[key] = (loaded_model, eval_env)
    return _WORKER_CACHE[key]


def _mc_worker(args):
    """
    Worker MC dla mp.Pool.
    args = (seed, policy_type, rule_name, T, model_path, vec_path)
    policy_type: "taylor" | "sac"
    rule_name: klucz do POLICY_CONFIGS (regula Taylora uzywana do tworzenia oczekiwan)
    """
    seed, policy_type, rule_name, T, model_path, vec_path = args
    rng = np.random.RandomState(seed)
    shocks = rng.normal(0, SIGMA_SHOCK, size=(T, 3))
    shocks[:, 2] = 0.0

    if policy_type == "taylor":
        return simulate_taylor_trajectory(rule_name, shocks, T)["loss"]
    else:
        loaded_model, eval_env = _get_or_load(model_path, vec_path)
        return simulate_sac_trajectory(loaded_model, eval_env, shocks, T, rule_name)["loss"]


def run_mc_evaluation(policies_to_run, n_runs=MC_N_RUNS, T=MC_T):
    """
    policies_to_run: slownik
        {display_name: (policy_type, rule_name, model_path, vec_path)}
    Zwraca: {display_name: np.array of shape (n_runs,) zawierajacy straty welfare}
    """
    all_args = []
    name_index = []

    for name, (ptype, rname, mpath, vpath) in policies_to_run.items():
        for run in range(n_runs):
            all_args.append((42 + run, ptype, rname, T, mpath, vpath))
            name_index.append(name)

    print(f"  Uruchamiam {len(all_args)} symulacji na {MC_WORKERS} workerach...")
    # maxtasksperchild=None — workery zywa do konca, dzieki czemu cache modeli
    # nie jest invalidowany.
    with mp.Pool(MC_WORKERS) as pool:
        flat_losses = pool.map(_mc_worker, all_args)

    results = {name: [] for name in policies_to_run}
    for name, loss in zip(name_index, flat_losses):
        results[name].append(loss)

    return {name: np.array(arr) for name, arr in results.items()}


# ==============================================================================
# 7. OBLICZANIE IRF (Impulse Response Function)
# ==============================================================================

def compute_taylor_irf(rule_name, shock_type="demand", T_irf=40, shock_size=1):
    """
    IRF: jeden szok w t=0, brak szokow dla t>0.
    shock_type: "demand" (eps[0]) lub "supply" (eps[1])
    """
    eps = np.zeros(3)
    if shock_type == "demand":
        eps[0] = shock_size
    else:
        eps[1] = shock_size

    pi, y, prev_r = 0.0, 0.0, 0.0
    hist_pi, hist_y, hist_r = [], [], []

    for t in range(T_irf):
        if t > 0:
            eps = P_MAT @ eps   # brak nowych szokow

        r, E_pi, E_y = compute_taylor_rate_and_expectations(
            rule_name, eps, prev_r)

        y_t  = E_y - (1.0 / SIGMA) * (r - E_pi) + eps[0]
        pi_t = BETA * E_pi + KAPPA * y_t + eps[1]

        hist_pi.append(pi_t)
        hist_y.append(y_t)
        hist_r.append(r)
        pi, y, prev_r = pi_t, y_t, r

    return {"pi": hist_pi, "y": hist_y, "r": hist_r}


def compute_sac_irf(model, eval_env, rule_name, shock_type="demand", T_irf=40, shock_size=1):
    """IRF dla agenta SAC wytrenowanego dla danej reguly (rule_name)."""
    eps = np.zeros(3)
    if shock_type == "demand":
        eps[0] = shock_size
    else:
        eps[1] = shock_size

    pi, y, prev_r = 0.0, 0.0, 0.0
    hist_pi, hist_y, hist_r = [], [], []

    for t in range(T_irf):
        if t > 0:
            eps = P_MAT @ eps

        state  = np.array([y, pi, eps[0], eps[1], eps[2], prev_r], dtype=np.float32)
        norm_s = eval_env.normalize_obs(state.reshape(1, -1))
        action, _ = model.predict(norm_s, deterministic=True)
        r = float(action[0][0])

        E_pi, E_y = compute_expectations(rule_name, eps, r)

        y_t  = E_y - (1.0 / SIGMA) * (r - E_pi) + eps[0]
        pi_t = BETA * E_pi + KAPPA * y_t + eps[1]

        hist_pi.append(pi_t)
        hist_y.append(y_t)
        hist_r.append(r)
        pi, y, prev_r = pi_t, y_t, r

    return {"pi": hist_pi, "y": hist_y, "r": hist_r}


# ==============================================================================
# 8. WYKRESY
#
# Wszystkie wykresy porownuja SAC z regula Taylora W OBREBIE tego samego
# srodowiska (tych samych macierzy BK). Kazde srodowisko to osobny panel/subplot.
# NIE mieszamy srodowisk na jednym wykresie — rozne macierze R/S generuja
# rozne dynamiki, wiec strata welfare z jednego srodowiska nie jest
# porownywalna ze strata z innego.
# ==============================================================================

def _short_name(name):
    return name.replace("\n", " ")


def _env_label(short):
    """Krotka etykieta srodowiska do tytulow subplot."""
    return {"simple": "Simple", "smoothing": "Smoothing", "forward": "Forward"}[short]


def plot_irf(irf_data, shock_type, save_path):
    """
    fig1/fig2: IRF — 3 kolumny (srodowiska) x 3 wiersze (pi, y, r).
    Kazda kolumna porownuje Taylor vs SAC w jednym srodowisku.
    """
    var_labels = {
        "pi": r"$\pi_t$",
        "y":  r"$\tilde{y}_t$",
        "r":  r"$i_t$"
    }
    shock_label = "Szok popytowy ($\\varepsilon_D$)" if shock_type == "demand" \
        else "Szok podazowy ($\\varepsilon_S$)"

    n_env = len(PAIRS)
    fig, axes = plt.subplots(3, n_env, figsize=(5 * n_env, 9), dpi=130,
                             sharey="row")

    for col, (short, tkey, skey) in enumerate(PAIRS):
        for row, var in enumerate(["pi", "y", "r"]):
            ax = axes[row, col]
            if tkey in irf_data:
                ax.plot(irf_data[tkey][var], label="Taylor",
                        color=COLOR_TAYLOR, linestyle="--", linewidth=1.5)
            if skey in irf_data:
                ax.plot(irf_data[skey][var], label="SAC",
                        color=COLOR_SAC, linestyle="-", linewidth=2.2)
            ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
            ax.grid(True, alpha=0.2)
            if col == 0:
                ax.set_ylabel(var_labels[var], fontsize=11)
            if row == 0:
                ax.set_title(_env_label(short), fontsize=11, fontweight="bold")
            if row == 2:
                ax.set_xlabel("Okresy", fontsize=9)
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(f"Impulse Response Function — {shock_label}\n"
                 f"Taylor vs SAC w kazdym srodowisku (osobne macierze BK)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Zapisano: {save_path}")


def plot_mc_boxplot(mc_results, save_path):
    """
    fig3: Violin + box — 3 panele, kazdy porownuje Taylor vs SAC
    w jednym srodowisku.
    """
    n_env = len(PAIRS)
    fig, axes = plt.subplots(1, n_env, figsize=(5 * n_env, 6), dpi=130,
                             sharey=True)

    for ax, (short, tkey, skey) in zip(axes, PAIRS):
        pair_data   = []
        pair_colors = []
        pair_labels = []
        if tkey in mc_results:
            pair_data.append(mc_results[tkey])
            pair_colors.append(COLOR_TAYLOR)
            pair_labels.append("Taylor")
        if skey in mc_results:
            pair_data.append(mc_results[skey])
            pair_colors.append(COLOR_SAC)
            pair_labels.append("SAC")

        if not pair_data:
            continue

        positions = list(range(len(pair_data)))
        parts = ax.violinplot(pair_data, positions=positions,
                              showmedians=True, showextrema=True, widths=0.6)
        for pc, c in zip(parts["bodies"], pair_colors):
            pc.set_facecolor(c)
            pc.set_alpha(0.5)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)
        for k in ("cmins", "cmaxes", "cbars"):
            parts[k].set_color("gray")

        ax.boxplot(pair_data, positions=positions, widths=0.25,
                   patch_artist=False, showfliers=False,
                   medianprops=dict(color="black", linewidth=2),
                   whiskerprops=dict(color="gray"),
                   capprops=dict(color="gray"),
                   boxprops=dict(color="black"))

        ax.set_xticks(positions)
        ax.set_xticklabels(pair_labels, fontsize=10)
        ax.set_title(_env_label(short), fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)

        # Procentowa roznica SAC vs Taylor
        if len(pair_data) == 2:
            t_mean = np.mean(pair_data[0])
            s_mean = np.mean(pair_data[1])
            rel = (t_mean - s_mean) / t_mean * 100
            sign = "+" if rel >= 0 else ""
            ax.text(0.5, 0.02, f"SAC: {sign}{rel:.1f}% vs Taylor",
                    transform=ax.transAxes, ha="center", fontsize=9,
                    color=COLOR_SAC if rel >= 0 else COLOR_TAYLOR,
                    fontweight="bold")

    axes[0].set_ylabel(
        r"Strata welfare  $\sum \beta^t (\pi_t^2 + \lambda_y \tilde{y}_t^2)$",
        fontsize=10)
    fig.suptitle(f"Rozklad welfare loss — Monte Carlo ({MC_N_RUNS} x {MC_T})\n"
                 f"Porownanie w obrebie kazdego srodowiska | Nizej = lepiej",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Zapisano: {save_path}")


def plot_welfare_bar(mc_results, save_path):
    """
    fig4: Slupki grupowane — dla kazdego srodowiska Taylor vs SAC obok siebie.
    Adnotacja: procentowa redukcja straty przez SAC.
    """
    n_env = len(PAIRS)
    bar_w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
    x = np.arange(n_env)

    t_means, t_sems = [], []
    s_means, s_sems = [], []
    env_labels = []

    for short, tkey, skey in PAIRS:
        env_labels.append(_env_label(short))
        if tkey in mc_results:
            arr = mc_results[tkey]
            t_means.append(np.mean(arr))
            t_sems.append(np.std(arr) / np.sqrt(len(arr)) * 1.96)
        else:
            t_means.append(0)
            t_sems.append(0)

        if skey in mc_results:
            arr = mc_results[skey]
            s_means.append(np.mean(arr))
            s_sems.append(np.std(arr) / np.sqrt(len(arr)) * 1.96)
        else:
            s_means.append(0)
            s_sems.append(0)

    ax.bar(x - bar_w / 2, t_means, bar_w, yerr=t_sems, capsize=4,
           color=COLOR_TAYLOR, alpha=0.8, label="Taylor",
           error_kw=dict(ecolor="black", elinewidth=1))
    ax.bar(x + bar_w / 2, s_means, bar_w, yerr=s_sems, capsize=4,
           color=COLOR_SAC, alpha=0.8, label="SAC",
           error_kw=dict(ecolor="black", elinewidth=1))

    # Adnotacja: % redukcja
    for i in range(n_env):
        if t_means[i] > 0 and s_means[i] > 0:
            rel = (t_means[i] - s_means[i]) / t_means[i] * 100
            sign = "+" if rel >= 0 else ""
            y_pos = max(t_means[i], s_means[i]) + max(t_sems[i], s_sems[i]) * 1.2
            ax.text(x[i], y_pos, f"{sign}{rel:.1f}%", ha="center", fontsize=9,
                    fontweight="bold",
                    color=COLOR_SAC if rel >= 0 else COLOR_TAYLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(env_labels, fontsize=11)
    ax.set_ylabel(r"Srednia strata welfare  $\bar{L}$", fontsize=11)
    ax.set_title("SAC vs Taylor — redukcja straty welfare\n"
                 f"(95% CI, {MC_N_RUNS} replikacji MC, nizej = lepiej)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Zapisano: {save_path}")


def plot_variance_frontier(var_data, save_path):
    """
    fig5: Granica wariancji — 3 panele, kazdy z para Taylor + SAC
    polaczonych strzalka (Taylor -> SAC).
    """
    n_env = len(PAIRS)
    fig, axes = plt.subplots(1, n_env, figsize=(5 * n_env, 5), dpi=130)

    for ax, (short, tkey, skey) in zip(axes, PAIRS):
        for name, marker, color, sz, lw in [
            (tkey, "o", COLOR_TAYLOR, 120, 1.5),
            (skey, "*", COLOR_SAC, 220, 2.0),
        ]:
            if name not in var_data:
                continue
            d = var_data[name]
            label = "Taylor" if name == tkey else "SAC"
            ax.scatter(d["y_var"], d["pi_var"], color=color, s=sz,
                       zorder=5, label=label, marker=marker, edgecolors="black",
                       linewidths=0.5)
            ax.annotate(label, (d["y_var"], d["pi_var"]),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=9, fontweight="bold" if name == skey else "normal")

        # Strzalka Taylor -> SAC
        if tkey in var_data and skey in var_data:
            td, sd = var_data[tkey], var_data[skey]
            ax.annotate("", xy=(sd["y_var"], sd["pi_var"]),
                        xytext=(td["y_var"], td["pi_var"]),
                        arrowprops=dict(arrowstyle="->", color="gray",
                                        lw=1.5, connectionstyle="arc3,rad=0.1"))

        ax.set_title(_env_label(short), fontsize=11, fontweight="bold")
        ax.set_xlabel(r"$\mathrm{Var}(\tilde{y}_t)$", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(r"$\mathrm{Var}(\pi_t)$", fontsize=10)
    fig.suptitle("Granica efektywnosci — wariancja inflacji vs luki produkcyjnej\n"
                 "(blizej (0,0) = lepsza polityka; strzalka: Taylor -> SAC)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Zapisano: {save_path}")


def plot_timeseries(ts_data, save_path):
    """
    fig6: Szeregi czasowe — 3 kolumny (srodowiska) x 3 wiersze (pi, y, r).
    Kazda kolumna: Taylor vs SAC w jednym srodowisku, wspolne szoki.
    """
    var_labels = {
        "pi": r"$\pi_t$",
        "y":  r"$\tilde{y}_t$",
        "r":  r"$i_t$",
    }
    n_env = len(PAIRS)
    fig, axes = plt.subplots(3, n_env, figsize=(5 * n_env, 10), dpi=130,
                             sharex=True, sharey="row")

    for col, (short, tkey, skey) in enumerate(PAIRS):
        for row, var in enumerate(["pi", "y", "r"]):
            ax = axes[row, col]
            if tkey in ts_data:
                ax.plot(ts_data[tkey][var], label="Taylor",
                        color=COLOR_TAYLOR, linestyle="--", linewidth=1.3,
                        alpha=0.9)
            if skey in ts_data:
                ax.plot(ts_data[skey][var], label="SAC",
                        color=COLOR_SAC, linestyle="-", linewidth=1.8,
                        alpha=0.9)
            ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
            ax.grid(True, alpha=0.2)
            if col == 0:
                ax.set_ylabel(var_labels[var], fontsize=11)
            if row == 0:
                ax.set_title(_env_label(short), fontsize=11, fontweight="bold")
            if row == 2:
                ax.set_xlabel("Okresy", fontsize=9)
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Szeregi czasowe — Taylor vs SAC w kazdym srodowisku\n"
                 "(wspolny scenariusz szokow, seed=2025)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Zapisano: {save_path}")


def plot_distributions(mc_series, save_path):
    """
    fig7: Gestosci pi_t i y_t — 3 kolumny (srodowiska) x 2 wiersze (pi, y).
    """
    from scipy.stats import gaussian_kde

    n_env = len(PAIRS)
    fig, axes = plt.subplots(2, n_env, figsize=(5 * n_env, 7), dpi=130)

    for col, (short, tkey, skey) in enumerate(PAIRS):
        for row, (var, xlabel) in enumerate([
            ("pi_all", r"$\pi_t$"),
            ("y_all",  r"$\tilde{y}_t$"),
        ]):
            ax = axes[row, col]
            for name, label, color, ls, lw in [
                (tkey, "Taylor", COLOR_TAYLOR, "--", 1.5),
                (skey, "SAC",    COLOR_SAC,    "-",  2.0),
            ]:
                if name not in mc_series:
                    continue
                vals = mc_series[name][var]
                if len(vals) < 2:
                    continue
                kde = gaussian_kde(vals, bw_method="silverman")
                xs  = np.linspace(vals.min(), vals.max(), 400)
                ax.plot(xs, kde(xs), label=label, color=color,
                        linestyle=ls, linewidth=lw)
            ax.axvline(0, color="black", linewidth=0.5, linestyle=":")
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=7, loc="upper right")
            if col == 0:
                ax.set_ylabel("Gestosc", fontsize=10)
            if row == 0:
                ax.set_title(_env_label(short), fontsize=11, fontweight="bold")
            if row == 1:
                ax.set_xlabel(xlabel, fontsize=10)
            else:
                ax.set_xlabel(xlabel, fontsize=10)

    fig.suptitle("Estymowane rozklady stacjonarne — Taylor vs SAC\n"
                 f"({MC_N_RUNS} replikacji x {MC_T} krokow)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Zapisano: {save_path}")


def plot_rate_smoothing(ts_data, save_path):
    """
    fig8: Zmiennosc stopy procentowej Std(delta_r) — slupki grupowane
    Taylor vs SAC w kazdym srodowisku.
    """
    n_env = len(PAIRS)
    bar_w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
    x = np.arange(n_env)

    t_stds, s_stds = [], []
    env_labels = []

    for short, tkey, skey in PAIRS:
        env_labels.append(_env_label(short))
        if tkey in ts_data:
            t_stds.append(np.std(np.diff(ts_data[tkey]["r"])))
        else:
            t_stds.append(0)
        if skey in ts_data:
            s_stds.append(np.std(np.diff(ts_data[skey]["r"])))
        else:
            s_stds.append(0)

    bars_t = ax.bar(x - bar_w / 2, t_stds, bar_w, color=COLOR_TAYLOR,
                    alpha=0.8, label="Taylor")
    bars_s = ax.bar(x + bar_w / 2, s_stds, bar_w, color=COLOR_SAC,
                    alpha=0.8, label="SAC")

    for i in range(n_env):
        for val, xpos in [(t_stds[i], x[i] - bar_w / 2),
                          (s_stds[i], x[i] + bar_w / 2)]:
            if val > 0:
                ax.text(xpos, val + max(max(t_stds), max(s_stds)) * 0.02,
                        f"{val:.5f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(env_labels, fontsize=11)
    ax.set_ylabel(r"$\mathrm{Std}(\Delta i_t)$", fontsize=11)
    ax.set_title("Zmiennosc stopy procentowej — Taylor vs SAC\n"
                 "(nizej = gladsza polityka)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Zapisano: {save_path}")


# ==============================================================================
# 8.5  SCENARIUSZE PROGNOSTYCZNE
#
# Prognozowanie warunkowe: dla zadanej sciezki innowacji eta_t (scenariusz),
# symulujemy trajektorie pi, y, r pod kazda polityka w kazdym srodowisku.
# To standardowe zastosowanie DSGE w bankach centralnych (por. NBP NECMOD,
# Bank of England COMPASS, FRBNY DSGE).
#
# Kazdy scenariusz to deterministyczna sekwencja innowacji (eta_t), ktore
# sa dodawane do AR(1) procesu szokow:  eps_t = P_mat @ eps_{t-1} + eta_t.
# ==============================================================================

T_FORECAST = 40   # horyzont prognozy (kwartaly)

# Amplitudy szokow scenariuszowych dobrane tak, by miescily sie w przedziale
# ~1-1.5 sigma_shock = 0.01-0.015. Wczesniejsze wartosci 0.03 (3 sigma) dawaly
# trajektorie obserwacji DALEKO poza dystrybucja, na ktorej VecNormalize zebral
# statystyki w treningu — siec neuronowa SAC ekstrapolowala wtedy do obszaru
# bez treningowego pokrycia, co oznacza nieprawidlowe wyniki ewaluacji
# out-of-distribution (OOD).
#
# Przy innowacjach <= 1.5 sigma scenariusze pozostaja w obszarze obserwowanym
# w treningu (sigma_shock = 0.01, wiec typowa innowacja w treningu ma amplitude
# do ~3 sigma w tail-events, ale srednio ~1 sigma).

def _scenario_demand_boom():
    """Utrzymujacy sie boom popytowy: dodatnie innowacje popytu przez 8 okresow (1 sigma)."""
    eta = np.zeros((T_FORECAST, 3))
    eta[0:8, 0] = 0.01
    return eta

def _scenario_supply_shock():
    """Szok kosztowy (np. ceny ropy): jednorazowy szok podazowy (1.5 sigma)."""
    eta = np.zeros((T_FORECAST, 3))
    eta[0, 1] = 0.015
    return eta

def _scenario_stagflation():
    """Stagflacja: jednoczesny szok podazowy (+) i szok popytowy (-) przez 5 okresow."""
    eta = np.zeros((T_FORECAST, 3))
    eta[0:5, 0] = -0.005
    eta[0:5, 1] =  0.01
    return eta

def _scenario_deep_recession():
    """Recesja z pozniejszym odbiciem: negatywny szok popytowy (1.5 sigma) + powrot."""
    eta = np.zeros((T_FORECAST, 3))
    eta[0, 0]    = -0.015
    eta[8:14, 0] =  0.005
    return eta

SCENARIOS = {
    "demand_boom":    ("Boom popytowy (8 kwartalow)",         _scenario_demand_boom),
    "supply_shock":   ("Szok kosztowy (jednorazowy)",         _scenario_supply_shock),
    "stagflation":    ("Stagflacja (podaz+, popyt-)",         _scenario_stagflation),
    "deep_recession": ("Recesja + odbicie (t=0, t=8..13)",    _scenario_deep_recession),
}


def simulate_scenario(rule_name, eta_path, agent=None, eval_env=None):
    """
    Uruchamia prognoze warunkowa na zadanej sciezce innowacji eta_path.
    Jesli agent=None — stosuje regule Taylora (rule_name).
    Jesli agent podany — stosuje polityke SAC w srodowisku rule_name.

    eta_path: (T, 3) — deterministyczne innowacje szokow (eta_D, eta_S, eta_R).
    """
    T = eta_path.shape[0]
    eps = np.zeros(3)
    pi, y, prev_r = 0.0, 0.0, 0.0
    hist = {"pi": [], "y": [], "r": []}
    cum_pi2, cum_y2, cum_dr2 = 0.0, 0.0, 0.0

    for t in range(T):
        eps = P_MAT @ eps + eta_path[t]

        if agent is None:
            r, E_pi, E_y = compute_taylor_rate_and_expectations(
                rule_name, eps, prev_r)
        else:
            state  = np.array([y, pi, eps[0], eps[1], eps[2], prev_r], dtype=np.float32)
            norm_s = eval_env.normalize_obs(state.reshape(1, -1))
            action, _ = agent.predict(norm_s, deterministic=True)
            r = float(action[0][0])
            E_pi, E_y = compute_expectations(rule_name, eps, r)

        y_t  = E_y - (1.0 / SIGMA) * (r - E_pi) + eps[0]
        pi_t = BETA * E_pi + KAPPA * y_t + eps[1]
        dr   = r - prev_r

        cum_pi2 += (BETA ** t) * pi_t**2
        cum_y2  += (BETA ** t) * y_t**2
        cum_dr2 += (BETA ** t) * dr**2

        hist["pi"].append(pi_t); hist["y"].append(y_t); hist["r"].append(r)
        pi, y, prev_r = pi_t, y_t, r

    # loss_core = czyste kryterium Woodforda (zgodne z MC w simulate_*_trajectory)
    # loss_full = z dodatkowa kara za zmiennosc stopy (= funkcja nagrody RL)
    # KRYTERIUM NORMATYWNE = loss_core. loss_full zachowane wylacznie do diagnostyki
    # (np. analizy, ile agent "placi" za reward shaping).
    loss_core = cum_pi2 + LAMBDA_Y * cum_y2
    loss_full = loss_core + 0.1 * cum_dr2
    return {
        "pi": hist["pi"], "y": hist["y"], "r": hist["r"],
        "loss_core": loss_core, "loss_full": loss_full,
        "cum_pi2": cum_pi2, "cum_y2": cum_y2, "cum_dr2": cum_dr2,
    }


def plot_scenario(scenario_key, scenario_results, eta_path, save_path):
    """
    Wykres scenariusza: 3 kolumny (srodowiska) x 3 wiersze (pi, y, r).
    Taylor czerwona przerywana, SAC zielona ciagla. Linia zerowa jako baseline.
    Pionowe linie pokazuja momenty, gdy wystepuja niezerowe innowacje.
    """
    title, _ = SCENARIOS[scenario_key]
    var_labels = {"pi": r"$\pi_t$", "y": r"$\tilde{y}_t$", "r": r"$i_t$"}
    active_steps = np.where(np.any(np.abs(eta_path) > 1e-12, axis=1))[0]

    n_env = len(PAIRS)
    fig, axes = plt.subplots(3, n_env, figsize=(5 * n_env, 9), dpi=130,
                             sharey="row")

    for col, (short, tkey, skey) in enumerate(PAIRS):
        t_res = scenario_results[tkey]
        s_res = scenario_results.get(skey)
        for row, var in enumerate(["pi", "y", "r"]):
            ax = axes[row, col]
            ax.plot(t_res[var], label="Taylor",
                    color=COLOR_TAYLOR, linestyle="--", linewidth=1.5)
            if s_res is not None:
                ax.plot(s_res[var], label="SAC",
                        color=COLOR_SAC, linestyle="-", linewidth=2.0)
            ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
            for s in active_steps:
                ax.axvline(s, color="gray", alpha=0.15, linewidth=0.5)
            ax.grid(True, alpha=0.2)
            if col == 0:
                ax.set_ylabel(var_labels[var], fontsize=11)
            if row == 0:
                ax.set_title(_env_label(short), fontsize=11, fontweight="bold")
            if row == 2:
                ax.set_xlabel("Okresy (kwartaly)", fontsize=9)
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(f"Scenariusz: {title}\n"
                 f"(szare linie = aktywne innowacje; Taylor vs SAC w kazdym srodowisku)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Zapisano: {save_path}")


def plot_scenario_welfare(welfare_table, save_path):
    """
    Zbiorczy wykres: dla kazdego scenariusza slupki pokazuja strate
    welfare Woodforda (pi^2 + lambda_y * y^2, bez kary za zmiennosc stopy)
    — Taylor vs SAC w 3 srodowiskach. Spojne z kryterium uzywanym w MC.
    """
    n_scn = len(SCENARIOS)
    fig, axes = plt.subplots(1, n_scn, figsize=(5 * n_scn, 5), dpi=130,
                             sharey=False)

    for ax, (sk, (title, _)) in zip(axes, SCENARIOS.items()):
        t_vals = [welfare_table[sk][tkey]  for _, tkey, _ in PAIRS]
        s_vals = [welfare_table[sk].get(skey, 0.0) for _, _, skey in PAIRS]
        x = np.arange(len(PAIRS)); w = 0.35
        bt = ax.bar(x - w/2, t_vals, w, color=COLOR_TAYLOR, alpha=0.8, label="Taylor")
        bs = ax.bar(x + w/2, s_vals, w, color=COLOR_SAC,    alpha=0.8, label="SAC")
        for i, (tv, sv) in enumerate(zip(t_vals, s_vals)):
            if tv > 0:
                pct = (tv - sv) / tv * 100
                ax.text(x[i], max(tv, sv) * 1.02, f"{pct:+.1f}%",
                        ha="center", fontsize=9, fontweight="bold",
                        color=("#27ae60" if sv < tv else "#e74c3c"))
        ax.set_xticks(x)
        ax.set_xticklabels([p[0] for p in PAIRS], fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(r"Strata welfare $\sum \beta^t (\pi_t^2 + \lambda_y \tilde{y}_t^2)$ — prognozy scenariuszowe"
                 "\nProcent: redukcja SAC vs Taylor (zielony = SAC lepszy)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Zapisano: {save_path}")


# ==============================================================================
# 9. MAIN
# ==============================================================================

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 70)
    print("  NK DSGE — Porownanie polityk monetarnych")
    print("  Taylor vs SAC w 3 srodowiskach (simple, smoothing, forward)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # A. Ladowanie 3 modeli SAC (po jednym na regule Taylora)
    # -------------------------------------------------------------------------
    sac_models = {}    # taylor_key -> (model, eval_env, model_path, vec_path)
    for short, tkey, skey in PAIRS:
        fname, vname = SAC_FILES[short]
        mpath = os.path.join(SAVE_DIR, fname)
        vpath = os.path.join(SAVE_DIR, vname)
        if os.path.exists(mpath + ".zip") and os.path.exists(vpath):
            m   = SAC.load(mpath)
            env = VecNormalize.load(vpath, DummyVecEnv([_make_env]))
            env.training    = False
            env.norm_reward = False
            sac_models[tkey] = (m, env, mpath, vpath)
            print(f"  [OK] SAC ({short}) wczytany: {fname}.zip")
        else:
            print(f"  [!!] Brak modelu SAC dla '{short}' — uruchom nk_dsge_sac_optimized.py")

    # -------------------------------------------------------------------------
    # B. Budowanie slownika polityk do MC
    #    Kluczem jest nazwa wyswietlana, wartoscia: (ptype, taylor_key, mpath, vpath)
    #    taylor_key sluzy do tworzenia oczekiwan (wskazuje POLICY_CONFIGS)
    # -------------------------------------------------------------------------
    policies = {}
    for short, tkey, skey in PAIRS:
        policies[tkey] = ("taylor", tkey, None, None)
        if tkey in sac_models:
            _, _, mpath, vpath = sac_models[tkey]
            policies[skey] = ("sac", tkey, mpath, vpath)

    print(f"\n  Polityki do ewaluacji: {[_short_name(n) for n in policies]}")

    # -------------------------------------------------------------------------
    # C. Monte Carlo evaluation
    # -------------------------------------------------------------------------
    print(f"\n[1/4] Monte Carlo ({MC_N_RUNS} replikacji x {MC_T} krokow)...")
    t0 = time.time()
    mc_losses = run_mc_evaluation(policies, n_runs=MC_N_RUNS, T=MC_T)
    mc_time = time.time() - t0

    print(f"\n  Czas MC: {mc_time:.1f}s")
    print(f"\n{'='*65}")
    print(f"  WYNIKI WELFARE LOSS  (nizej = lepiej)")
    print(f"{'='*65}")
    for short, tkey, skey in PAIRS:
        t_loss = np.mean(mc_losses.get(tkey, [np.nan]))
        print(f"\n  --- {_env_label(short)} ---")
        print(f"    Taylor: {t_loss:.6f}  +/- {np.std(mc_losses.get(tkey, [0])):.6f}")
        if skey in mc_losses:
            s_loss = np.mean(mc_losses[skey])
            rel = (t_loss - s_loss) / t_loss * 100
            print(f"    SAC:    {s_loss:.6f}  +/- {np.std(mc_losses[skey]):.6f}"
                  f"   ({rel:+.1f}% vs Taylor)")
        else:
            print(f"    SAC:    brak modelu")
    print(f"\n{'='*65}")

    # -------------------------------------------------------------------------
    # D. Trajektorie do wykresow
    #    (jedno losowe ziarno, wspolne szoki dla wszystkich polityk w danym
    #     srodowisku — szoki sa takie same dla Taylor i SAC)
    # -------------------------------------------------------------------------
    print("\n[2/4] Generowanie trajektorii do wykresow...")
    np.random.seed(2025)
    T_plot = 200
    shared_shocks = np.random.normal(0, SIGMA_SHOCK, (T_plot, 3))
    shared_shocks[:, 2] = 0.0

    ts_results = {}
    mc_series  = {}
    var_data   = {}

    for name, (ptype, rname, mpath, vpath) in policies.items():
        if ptype == "taylor":
            ts = simulate_taylor_trajectory(rname, shared_shocks, T_plot)
        else:
            m, env, _, _ = sac_models[rname]
            ts = simulate_sac_trajectory(m, env, shared_shocks, T_plot, rname)
        ts_results[name] = ts

        all_pi, all_y = [], []
        for s in range(20):
            rng_s = np.random.RandomState(s * 13)
            sh_s  = rng_s.normal(0, SIGMA_SHOCK, (T_plot, 3))
            sh_s[:, 2] = 0.0
            if ptype == "taylor":
                traj = simulate_taylor_trajectory(rname, sh_s, T_plot)
            else:
                m, env, _, _ = sac_models[rname]
                traj = simulate_sac_trajectory(m, env, sh_s, T_plot, rname)
            all_pi.extend(traj["pi"])
            all_y.extend(traj["y"])

        mc_series[name] = {
            "pi_all": np.array(all_pi),
            "y_all":  np.array(all_y),
        }
        var_data[name] = {
            "pi_var": np.var(all_pi),
            "y_var":  np.var(all_y),
            "r_var":  np.var(ts["r"]),
        }

    # -------------------------------------------------------------------------
    # E. IRF
    # -------------------------------------------------------------------------
    print("\n[3/4] Obliczanie IRF...")
    irf_demand = {}
    irf_supply = {}

    for name, (ptype, rname, mpath, vpath) in policies.items():
        if ptype == "taylor":
            irf_demand[name] = compute_taylor_irf(rname, "demand", T_irf=40)
            irf_supply[name] = compute_taylor_irf(rname, "supply", T_irf=40)
        else:
            m, env, _, _ = sac_models[rname]
            irf_demand[name] = compute_sac_irf(m, env, rname, "demand", T_irf=40)
            irf_supply[name] = compute_sac_irf(m, env, rname, "supply", T_irf=40)

    # -------------------------------------------------------------------------
    # F. Rysowanie wykresow
    # -------------------------------------------------------------------------
    print("\n[4/4] Zapisywanie wykresow...")

    plot_irf(irf_demand, "demand",
             os.path.join(SAVE_DIR, "fig1_irf_demand.png"))

    plot_irf(irf_supply, "supply",
             os.path.join(SAVE_DIR, "fig2_irf_supply.png"))

    plot_mc_boxplot(mc_losses,
                    os.path.join(SAVE_DIR, "fig3_mc_boxplot.png"))

    plot_welfare_bar(mc_losses,
                     os.path.join(SAVE_DIR, "fig4_welfare_bar.png"))

    plot_variance_frontier(var_data,
                           os.path.join(SAVE_DIR, "fig5_variance_frontier.png"))

    plot_timeseries(ts_results,
                    os.path.join(SAVE_DIR, "fig6_timeseries.png"))

    plot_distributions(mc_series,
                       os.path.join(SAVE_DIR, "fig7_mc_distributions.png"))

    plot_rate_smoothing(ts_results,
                        os.path.join(SAVE_DIR, "fig8_rate_smoothing.png"))

    # -------------------------------------------------------------------------
    # G. Prognozy scenariuszowe (warunkowe forecasting — Taylor vs SAC)
    # -------------------------------------------------------------------------
    print("\n[5/5] Prognozy scenariuszowe (warunkowe)...")
    welfare_table = {}
    scenario_idx  = 9
    for sk, (title, build) in SCENARIOS.items():
        eta_path = build()
        results  = {}
        for short, tkey, skey in PAIRS:
            results[tkey] = simulate_scenario(tkey, eta_path, agent=None)
            if tkey in sac_models:
                m, env, _, _ = sac_models[tkey]
                results[skey] = simulate_scenario(tkey, eta_path, agent=m, eval_env=env)
        plot_scenario(sk, results, eta_path,
                      os.path.join(SAVE_DIR, f"fig{scenario_idx}_scenario_{sk}.png"))
        # Kryterium normatywne = czysta strata Woodforda (loss_core), spojnie z MC.
        # Kara 0.1*(delta_r)^2 z funkcji nagrody RL nie wchodzi do oceny welfare.
        welfare_table[sk] = {name: r["loss_core"] for name, r in results.items()}
        scenario_idx += 1

    plot_scenario_welfare(welfare_table,
                          os.path.join(SAVE_DIR, "fig13_scenario_welfare.png"))

    print(f"\n{'='*70}")
    print("  PROGNOZY SCENARIUSZOWE — Strata welfare Woodforda (loss_core)")
    print(f"{'='*70}")
    for sk, (title, _) in SCENARIOS.items():
        print(f"\n  {title}")
        for short, tkey, skey in PAIRS:
            t_loss = welfare_table[sk][tkey]
            if skey in welfare_table[sk]:
                s_loss = welfare_table[sk][skey]
                rel    = (t_loss - s_loss) / t_loss * 100 if t_loss > 0 else 0.0
                print(f"    {short:10s}  Taylor: {t_loss:8.4f}   SAC: {s_loss:8.4f}   ({rel:+6.1f}%)")
            else:
                print(f"    {short:10s}  Taylor: {t_loss:8.4f}   SAC: brak")

    print(f"\n{'='*70}")
    print(f"  Wszystkie wykresy zapisano do: {SAVE_DIR}/")
    print(f"{'='*70}")
