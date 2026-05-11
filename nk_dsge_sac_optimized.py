# ==============================================================================
# NK DSGE + SAC: Trening 3 agentow — po jednym na kazda regule Taylora
#
# HARDWARE-TARGET:
#   CPU:  AMD Ryzen 5 3600, 6 rdzeni / 12 watkow
#   RAM:  32 GB
#   GPU:  brak (ROCm niedostepny na Windows) -> CPU-only
#
# IDEA:
#   Agent zastepuje regule Taylora, ale sektor prywatny (gospodarstwa i firmy)
#   formuje oczekiwania zgodnie z ZAKLADANA regula polityki pienieznej.
#   Dlatego macierze BK (R_mat, P_mat, S_mat) zaleza od uzytej reguly —
#   kazda regula => inne oczekiwania => inne srodowisko RL.
#
#   Trenujemy trzech agentow:
#     1. SAC-Simple    — oczekiwania jak pod prosta regula Taylora
#     2. SAC-Smoothing — oczekiwania jak pod regula z wygladzaniem (stan rozszerzony o r_{t-1})
#     3. SAC-Forward   — oczekiwania jak pod regula forward-looking
#
#   Dla kazdego agenta uruchamiamy osobno:
#     Faza 1: Optuna (30 triali)
#     Faza 2: Finalny trening (300k krokow)
#     Faza 3: Zapis modelu + VecNormalize
#
#   Ewaluacja i porownanie: nk_dsge_comparison.py
#
# REFERENCJE:
#   Haarnoja et al. (2018)    — SAC algorithm
#   Hinterlang & Tanzer (2021) — RL pod zalozeniem oczekiwan z reguly Taylora
#   Poutineau et al. (2015)   — kalibracja NK3
# ==============================================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import optuna
import torch
import os
import time
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback


# ==============================================================================
# 0. KONFIGURACJA HARDWARE
# ==============================================================================

N_PHYSICAL_CORES = 6
OPTUNA_N_JOBS = 6
TORCH_THREADS_PER_TRIAL = 2     # 6 x 2 = 12
TORCH_THREADS_FINAL = 6


# ==============================================================================
# 1. MACIERZE BLANCHARDA-KAHNA DLA KAZDEJ REGULY TAYLORA
#
# Duplikat wartosci z nk_dsge_comparison.py. Obie kopie MUSZA byc identyczne,
# bo srodowisko treningowe i ewaluacyjne musza uzywac tych samych macierzy.
# Wartosci pochodza z rozwiazania BK w dsge.R (gEcon) dla kazdej reguly osobno.
# ==============================================================================

# --- Prosta regula Taylora: r_t = 1.5*pi + 0.125*y + eps_R ---
# Stan BK: [eps_D, eps_S, eps_R] (3-wymiarowy)
R_MAT_SIMPLE = np.array([
    [1.2115,  1.5879, -0.1096],   # wiersz 0: E_t{pi_{t+1}}
    [1.9134,  1.8526,  0.1875],   # wiersz 1: E_t{r_{t+1}}
    [0.7693, -4.2345, -0.3855],   # wiersz 2: E_t{y_{t+1}}
])

# --- Regula z wygladzaniem: r_t = 0.8*r_{t-1} + 0.2*(1.5*pi + 0.125*y) + eps_R ---
# Stan BK: [eps_D, eps_S, eps_R, r_{t-1}] (4-wymiarowy — r_{t-1} jest stanem)
R_MAT_SMOOTH = np.array([
    [1.4692,  1.8374, -0.6292, -0.7203],   # wiersz 0: E_t{pi_{t+1}}
    [3.0839, -1.9934, -1.4911, -1.9755],   # wiersz 1: E_t{y_{t+1}}
])

# --- Forward-looking: r_t = 1.5*E_t{pi_{t+1}} + 0.125*E_t{y_{t+1}} + eps_R ---
# Stan BK: [eps_D, eps_S, eps_R] (3-wymiarowy)
R_MAT_FORWARD = np.array([
    [1.5387,  1.9046, -0.1608],   # wiersz 0: E_t{pi_{t+1}}
    [2.1871,  2.1175,  0.2752],   # wiersz 1: E_t{r_{t+1}}
    [0.9770, -4.0334, -0.5659],   # wiersz 2: E_t{y_{t+1}}
])

# Macierz trwalosci szokow AR(1) — wspolna dla wszystkich regul
P_MAT = np.diag([0.90, 0.90, 0.40])   # [rho_D, rho_S, rho_R]


# ==============================================================================
# 2. KONFIGURACJA REGUL (wiersze R_mat, wymiar stanu BK)
# ==============================================================================

RULE_CONFIGS = {
    "simple": {
        "R_mat":          R_MAT_SIMPLE,
        "E_pi_row":       0,
        "E_y_row":        2,
        "expanded_state": False,   # stan BK = [eps_D, eps_S, eps_R]
    },
    "smoothing": {
        "R_mat":          R_MAT_SMOOTH,
        "E_pi_row":       0,
        "E_y_row":        1,       # w wersji smooth wiersz 1 = y (r jest stanem)
        "expanded_state": True,    # stan BK = [eps_D, eps_S, eps_R, r_{t-1}]
    },
    "forward": {
        "R_mat":          R_MAT_FORWARD,
        "E_pi_row":       0,
        "E_y_row":        2,
        "expanded_state": False,
    },
}


# ==============================================================================
# 3. SRODOWISKO NK DSGE — parametryzowane regula Taylora
# ==============================================================================

class NewKeynesianEnv(gym.Env):
    """
    Srodowisko Gymnasium: 3-rownaniowy NK DSGE.
    Agent = bank centralny, wybiera r_t.

    Rownania strukturalne (zawsze te same):
      y_t  = E_t{y_{t+1}}  - (1/sigma)*(r_t - E_t{pi_{t+1}}) + eps_D_t    (DIS)
      pi_t = beta*E_t{pi_{t+1}} + kappa*y_t + eps_S_t                      (NKPC)

    Oczekiwania E_t{y_{t+1}}, E_t{pi_{t+1}} zaleza od rule_config:
      E_t{pi_{t+1}} = R_mat[E_pi_row] @ state_bk
      E_t{y_{t+1}}  = R_mat[E_y_row]  @ state_bk
    gdzie state_bk = [eps_D, eps_S, eps_R] lub [eps_D, eps_S, eps_R, r_{t-1}]
    (zaleznie od 'expanded_state' w rule_config).

    Obserwacja (6-wymiarowa, ta sama dla kazdej reguly):
      [y_{t-1}, pi_{t-1}, eps_D_t, eps_S_t, eps_R_t, r_{t-1}]
    """

    metadata = {"render_modes": []}

    def __init__(self, rule_config):
        super().__init__()

        self.beta  = 0.99
        self.sigma = 1.0
        self.phi   = 1.0
        self.theta = 0.75
        self.kappa = ((1 - self.theta) * (1 - self.theta * self.beta)
                      / self.theta) * (self.sigma + self.phi)
        self.sigma_shock = 0.01
        self.lambda_y    = self.kappa / 6.0

        # Macierze i ustawienia specyficzne dla reguly
        self.R_mat          = rule_config["R_mat"]
        self.E_pi_row       = rule_config["E_pi_row"]
        self.E_y_row        = rule_config["E_y_row"]
        self.expanded_state = rule_config["expanded_state"]
        self.P_mat          = P_MAT

        self.action_space = spaces.Box(
            low=np.array([-5.0], dtype=np.float32),
            high=np.array([5.0], dtype=np.float32),
            dtype=np.float32,
        )
        high_obs = np.array([10.0, 10.0, 3.0, 3.0, 3.0, 5.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high_obs, high=high_obs, dtype=np.float32,
        )
        self._max_episode_steps = 200

    def _compute_expectations(self, r_current):
        """
        Oczekiwania sektora prywatnego uwarunkowane biezacym stanem.

        W konwencji gEcon: E_t{c_{t+1}} = R @ s_t, gdzie dla modelu smoothing
        s_t = [eps_D_t, eps_S_t, eps_R_t, r_t].  Czwarty element to BIEZACA
        stopa r_t (ktora stanie sie predetermined w t+1), NIE r_{t-1}.
        """
        if self.expanded_state:
            state_bk = np.concatenate([self.eps, [r_current]])
        else:
            state_bk = self.eps
        E_pi = float(self.R_mat[self.E_pi_row] @ state_bk)
        E_y  = float(self.R_mat[self.E_y_row]  @ state_bk)
        return E_pi, E_y

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.eps    = np.zeros(3)
        self.pi     = 0.0
        self.y      = 0.0
        self.prev_r = 0.0
        self.steps  = 0
        # Losuj szoki pierwszego okresu, zeby agent widzial eps_1 przed decyzja r_1.
        # eta[2] (innowacja szoku polityki monetarnej) jest zerowana, bo agent SAC
        # CALKOWICIE zastepuje regule polityki — szok MP to "exogenous error in the
        # rule", a skoro nie ma reguly, nie ma szoku. Slot eps[2] pozostaje
        # w obserwacji dla spojnosci wymiarowej z modelem DSGE; jest zawsze zerem.
        eta = self.np_random.normal(0, self.sigma_shock, size=3)
        eta[2] = 0.0
        self.eps = self.P_mat @ self.eps + eta
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array(
            [self.y, self.pi, self.eps[0], self.eps[1], self.eps[2], self.prev_r],
            dtype=np.float32,
        )

    def step(self, action):
        # SB3 SAC z action_space=Box(-5, 5) zwraca akcje juz w przedziale,
        # bo wewnetrznie skaluje tanh-squashed sample. Dodatkowy np.clip nie jest
        # potrzebny i tworzy plaski obszar gradientu na granicy.
        r_agent = float(action[0])

        # Oczekiwania uzycia r_t (akcji agenta) — nie r_{t-1} (self.prev_r).
        # W gEcon: E_t{c_{t+1}} = R @ s_t, a s_t zawiera r_t.
        E_pi, E_y = self._compute_expectations(r_agent)

        y_t  = E_y - (1.0 / self.sigma) * (r_agent - E_pi) + self.eps[0]
        pi_t = self.beta * E_pi + self.kappa * y_t + self.eps[1]

        delta_r = r_agent - self.prev_r
        reward = -(pi_t**2 + self.lambda_y * y_t**2 + 0.1 * delta_r**2)

        self.pi     = pi_t
        self.y      = y_t
        self.prev_r = r_agent
        self.steps += 1

        terminated = bool(abs(self.y) > 10.0 or abs(self.pi) > 10.0)
        truncated  = bool(self.steps >= self._max_episode_steps)
        if terminated:
            reward -= 500.0

        # Losuj szoki nastepnego okresu — trafia do obserwacji kolejnego kroku
        eta = self.np_random.normal(0, self.sigma_shock, size=3)
        eta[2] = 0.0
        self.eps = self.P_mat @ self.eps + eta

        info = {
            "output_gap":    float(self.y),
            "inflation":     float(self.pi),
            "interest_rate": float(r_agent),
        }
        return self._get_obs(), reward, terminated, truncated, info


def make_env_factory(rule_name):
    """Zwraca fabryke srodowisk dla danej reguly (uzywana w DummyVecEnv)."""
    cfg = RULE_CONFIGS[rule_name]
    def _make():
        return NewKeynesianEnv(cfg)
    return _make


# ==============================================================================
# 4. CALLBACK OPTUNA
# ==============================================================================

class TrialEvalCallback(EvalCallback):
    def __init__(self, eval_env, trial, n_eval_episodes=5,
                 eval_freq=5000, deterministic=True, verbose=0):
        super().__init__(
            eval_env, n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq, deterministic=deterministic, verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


# ==============================================================================
# 5. FUNKCJA CELU OPTUNA — SAC
# ==============================================================================

def objective(trial, rule_name):
    """SAC trial: 25k krokow w srodowisku dla danej reguly."""
    torch.set_num_threads(TORCH_THREADS_PER_TRIAL)

    learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True)
    gamma         = trial.suggest_float("gamma", 0.95, 0.9999)
    tau           = trial.suggest_float("tau", 0.001, 0.05, log=True)
    batch_size    = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    # Pelny zakres bufora — w treningu finalnym uzywamy DOKLADNIE tej wartosci,
    # ktora wybrala Optuna, zeby uniknac niespojnosci hiperparametrow.
    buffer_size   = trial.suggest_categorical(
        "buffer_size", [5_000, 10_000, 25_000, 50_000, 100_000, 200_000]
    )
    use_auto_ent  = trial.suggest_categorical("use_auto_ent", [True, False])
    ent_coef      = "auto" if use_auto_ent else trial.suggest_float(
        "ent_coef_value", 0.01, 0.3, log=True
    )
    net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    net_arch_map  = {"small": [64, 64], "medium": [128, 128], "large": [256, 256]}
    net_arch      = net_arch_map[net_arch_type]

    mk = make_env_factory(rule_name)
    env      = VecNormalize(DummyVecEnv([mk]), norm_obs=True, norm_reward=True)
    eval_env = VecNormalize(
        DummyVecEnv([mk]), norm_obs=True, norm_reward=False, training=False
    )

    model = SAC(
        "MlpPolicy", env,
        learning_rate=learning_rate, gamma=gamma, tau=tau,
        batch_size=batch_size, buffer_size=buffer_size,
        ent_coef=ent_coef,
        train_freq=4, gradient_steps=1,
        policy_kwargs=dict(net_arch=net_arch),
        device="cpu", verbose=0,
    )

    eval_cb = TrialEvalCallback(eval_env, trial, n_eval_episodes=3, eval_freq=5_000)

    try:
        model.learn(total_timesteps=25_000, callback=eval_cb)
    except (AssertionError, ValueError):
        env.close(); eval_env.close()
        return -float("inf")

    env.close(); eval_env.close()
    if eval_cb.is_pruned:
        raise optuna.exceptions.TrialPruned()
    return eval_cb.last_mean_reward


# ==============================================================================
# 6. TRENING FINALNY
# ==============================================================================

def train_final_model(best_params, rule_name, total_timesteps=300_000):
    torch.set_num_threads(TORCH_THREADS_FINAL)

    params = best_params.copy()
    net_arch_type = params.pop("net_arch")
    net_arch = {"small": [64, 64], "medium": [128, 128], "large": [256, 256]}[net_arch_type]

    use_auto_ent = params.pop("use_auto_ent")
    if use_auto_ent:
        ent_coef = "auto"
        params.pop("ent_coef_value", None)
    else:
        ent_coef = params.pop("ent_coef_value")

    # Trening finalny uzywa DOKLADNIE tej wartosci buffer_size, ktora wybrala
    # Optuna. Wczesniej byla tu twardo zakodowana wartosc 200k, co unieważniało
    # caly proces strojenia (pozostale hiperparametry byly optymalne dla mniejszego
    # bufora). Teraz Optuna stroi buffer_size w pelnym zakresie [5k, 200k].
    env = VecNormalize(
        DummyVecEnv([make_env_factory(rule_name)]),
        norm_obs=True, norm_reward=True,
    )

    model = SAC(
        "MlpPolicy", env,
        ent_coef=ent_coef,
        train_freq=4, gradient_steps=1,
        policy_kwargs=dict(net_arch=net_arch),
        **params, device="cpu", verbose=1,
    )

    print(f"\n{'='*60}")
    print(f"Trening finalny: SAC-{rule_name}, {total_timesteps} krokow")
    print(f"PyTorch watki: {TORCH_THREADS_FINAL}")
    print(f"{'='*60}")

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps)
    elapsed = time.time() - t0
    print(f"Czas treningu ({rule_name}): {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return model, env


# ==============================================================================
# 7. MAIN — petla po 3 regulach
# ==============================================================================

def train_agent_for_rule(rule_name, n_trials=30, total_timesteps=300_000, save_dir="saved_models"):
    """Pelen pipeline: Optuna + trening + zapis dla jednej reguly."""
    print("\n" + "#" * 70)
    print(f"#  AGENT SAC dla reguly: {rule_name.upper()}")
    print("#" * 70)

    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2),
    )

    print(f"\nOptuna ({rule_name}): {n_trials} triali, {OPTUNA_N_JOBS} rownoleglych")
    t0 = time.time()
    try:
        import tqdm  # noqa: F401
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    study.optimize(
        lambda trial: objective(trial, rule_name),
        n_trials=n_trials, n_jobs=OPTUNA_N_JOBS,
        show_progress_bar=has_tqdm,
    )
    elapsed = time.time() - t0

    print(f"\nNajlepsza nagroda ({rule_name}): {study.best_value:.4f}")
    print(f"Czas Optuna ({rule_name}): {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Najlepsze hiperparametry ({rule_name}):")
    for k, v in study.best_params.items():
        print(f"  {k:20s}: {v}")

    final_model, final_env = train_final_model(
        study.best_params, rule_name, total_timesteps=total_timesteps
    )

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"sac_{rule_name}")
    vec_path   = os.path.join(save_dir, f"vec_normalize_{rule_name}.pkl")
    final_model.save(model_path)
    final_env.save(vec_path)
    print(f"Zapisano model:      {model_path}.zip")
    print(f"Zapisano normalizer: {vec_path}")

    return model_path, vec_path


if __name__ == "__main__":
    print("=" * 70)
    print("  NK DSGE + SAC — TRENING 3 AGENTOW (po 1 na kazda regule)")
    print("  Hardware: AMD Ryzen 5 3600 (6C/12T) | 32 GB RAM | CPU-only")
    print("=" * 70)
    print()
    print("  Reguly: simple, smoothing, forward")
    print("  Dla kazdej: Optuna (30 triali x 25k krokow) + trening finalny (300k krokow)")
    print("  Szacowany czas calkowity: ~75-120 min")
    print("  (rozszerzona siatka buffer_size do 200k => wolniejsze triale Optuny)")
    print()

    t_start = time.time()

    for rule_name in ["simple", "smoothing", "forward"]:
        train_agent_for_rule(
            rule_name,
            n_trials=30,
            total_timesteps=300_000,
            save_dir="saved_models",
        )

    total_elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"  Trening 3 agentow zakonczony.")
    print(f"  Czas calkowity: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print("  Ewaluacja: uruchom nk_dsge_comparison.py")
    print("=" * 70)
