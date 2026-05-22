"""
Restricted Boltzmann Machine — merged implementation.

Core algorithm from rbm_model_pro.py (Hinton-faithful):
  · SGD with momentum schedule (0.5 → final after ramp)
  · LR schedule: 'constant', 'hinton' (warm-up), 'linear'
  · Persistent Contrastive Divergence (PCD)
  · Proper Ising model (tanh) for spins
  · Potts hidden units (softmax one-hot)
  · Weight decay (L2) + L1 penalty (gamma)

Infrastructure:
  · train() returns structured results dict
  · Full history & metrics tracking (configurable frequency)
  · Free energy, reconstruction error, pseudo-likelihood
  · Gibbs chain sampling, joint energy
  · npz/json persistence (save_results / load_results)

Results dict is compatible with rbm_plots.py.

Typical usage
-------------
>>> from rbm import RBM, load_results
>>> model = RBM(n_v=784, n_h=24, optimizer='sgd', lr=0.01)
>>> results = model.train(data, Nepoch=150, save_dir='run_01')
>>> # results["config"]  results["weights"]  results["history"]  results["metrics"]
>>>
>>> results = load_results('run_01')   # reload in another session
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import logit


# ═════════════════════════════════════════════════════════════════════════
#  RBM
# ═════════════════════════════════════════════════════════════════════════

class RBM:
    """
    Restricted Boltzmann Machine.

    Parameters (model + optimiser — set once)
    ------------------------------------------
    n_v, n_h : int
        Visible / hidden units.
    optimizer : ``'sgd'`` | ``'rmsprop'``
    lr : float
        Base learning rate.
    lr_schedule : ``'constant'`` | ``'hinton'`` | ``'linear'``
        ``'hinton'`` → lr = 0.1 for the first *lr_ramp* epochs, then *lr*.
        ``'linear'`` → ramps from *lr* to *lr_fin* (set in ``train()``).
    lr_ramp : int
        Warm-up epochs for ``'hinton'`` schedule (default 5).
    momentum : float
        Final SGD momentum.
    momentum_initial : float
        SGD momentum during the first *momentum_ramp* epochs.
    momentum_ramp : int
        Epochs before switching to final momentum (default 5).
    weight_decay : float
        L2 penalty on W.
    gamma : float
        L1 penalty on W.
    rmsprop_rho, rmsprop_eps : float
        RMSprop decay and epsilon.
    spins : bool
        ±1 encoding (Ising model with tanh activation).
    potts : bool
        One-hot hidden units (softmax sampling).
    w_init_scale : float
        Std of initial weight distribution.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        n_v: int,
        n_h: int,
        *,
        optimizer: str = "sgd",
        lr: float = 0.01,
        lr_schedule: str = "constant",
        lr_ramp: int = 5,
        momentum: float = 0.9,
        momentum_initial: float = 0.5,
        momentum_ramp: int = 5,
        weight_decay: float = 0.0002,
        gamma: float = 0.0,
        rmsprop_rho: float = 0.9,
        rmsprop_eps: float = 1e-8,
        spins: bool = False,
        potts: bool = False,
        w_init_scale: float = 0.01,
        seed: int = 12345,
    ):
        self.n_v = n_v
        self.n_h = n_h
        self.seed = seed

        # Optimiser
        self.optimizer_type = optimizer.lower()
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.lr_ramp = lr_ramp
        self.momentum_final = momentum
        self.momentum_initial = momentum_initial
        self.momentum_ramp = momentum_ramp
        self.wd = weight_decay
        self.gamma = gamma
        self.rho = rmsprop_rho
        self.eps = rmsprop_eps

        # Encoding
        self.spins = spins
        self.potts = potts

        # Weights
        np.random.seed(seed)
        self.W = np.random.normal(0, w_init_scale, (n_v, n_h))
        self.v_bias = np.zeros(n_v)
        self.h_bias = np.zeros(n_h)

        # Momentum buffers (SGD)
        self._vel_W = np.zeros_like(self.W)
        self._vel_vb = np.zeros_like(self.v_bias)
        self._vel_hb = np.zeros_like(self.h_bias)

        # Running squared gradients (RMSprop)
        self._sq_W = np.zeros_like(self.W)
        self._sq_vb = np.zeros_like(self.v_bias)
        self._sq_hb = np.zeros_like(self.h_bias)

        # PCD state
        self._persistent_v = None

    # ─────────────────────────────────────────────────────────────────────
    #  Core sampling  (batched — v has shape (N, n_v))
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))

    def sample_h(self, v: np.ndarray):
        """
        P(h | v).  Returns (prob, state).

        · default: sigmoid → Bernoulli
        · spins:   tanh → ±1
        · potts:   softmax → one-hot
        """
        act = v @ self.W + self.h_bias

        if self.spins:
            prob_spin = np.tanh(np.clip(act, -80, 80))
            prob_01 = (prob_spin + 1.0) / 2.0
            state = 2.0 * (prob_01 > np.random.rand(*prob_01.shape)).astype(float) - 1.0
            return prob_spin, state
    
        if self.potts:
            act_stable = act - act.max(axis=1, keepdims=True)
            e = np.exp(act_stable)
            prob = e / e.sum(axis=1, keepdims=True)
            state = np.zeros_like(prob)
            for i in range(v.shape[0]):
                state[i, np.random.choice(self.n_h, p=prob[i])] = 1.0
            return prob, state

        prob = self._sigmoid(act)
        state = (prob > np.random.rand(*prob.shape)).astype(float)
        return prob, state

    def sample_v(self, h: np.ndarray):
        """
        P(v | h).  Returns (prob, state).

        · default: sigmoid → Bernoulli {0,1}
        · spins:   tanh → ±1  (Ising visible units)
        """
        act = h @ self.W.T + self.v_bias

        if self.spins:
            prob_spin = np.tanh(np.clip(act, -80, 80))
            prob_01 = (prob_spin + 1.0) / 2.0
            state = 2.0 * (prob_01 > np.random.rand(*prob_01.shape)).astype(float) - 1.0
            return prob_spin, state

        prob = self._sigmoid(act)
        state = (prob > np.random.rand(*prob.shape)).astype(float)
        return prob, state

    # ─────────────────────────────────────────────────────────────────────
    #  Metrics
    # ─────────────────────────────────────────────────────────────────────

    def _free_energy_batch(self, v: np.ndarray) -> np.ndarray:
        """Per-sample free energy → (N,)."""
        wx_b = v @ self.W + self.h_bias
        vis_term = -v @ self.v_bias

        if self.potts:
            # Single softmax group: −log(Σ_k exp(x_k))  (logsumexp)
            max_wx = wx_b.max(axis=1, keepdims=True)
            hid_term = -(max_wx.squeeze(axis=1)
                         + np.log(np.sum(np.exp(wx_b - max_wx), axis=1)))
        elif self.spins:
            # ±1 hidden: −Σ_j log(2·cosh(x_j))
            # Stable form: log(2·cosh(x)) = |x| + log(1 + exp(−2|x|))
            abs_wx = np.abs(wx_b)
            hid_term = -np.sum(abs_wx + np.log1p(np.exp(-2.0 * abs_wx)),
                               axis=1)
        else:
            # Bernoulli {0,1}: −Σ_j log(1 + exp(x_j))
            hid_term = -np.sum(
                np.log1p(np.exp(np.clip(wx_b, -80, 80))), axis=1)

        return vis_term + hid_term

    def free_energy(self, v: np.ndarray) -> float:
        """Mean free energy over batch v."""
        return float(np.mean(self._free_energy_batch(v)))

    def reconstruction_error(self, X: np.ndarray) -> float:
        """Mean-field reconstruction MSE."""
        h_prob, _ = self.sample_h(X)
        v_recon, _ = self.sample_v(h_prob)
        return float(np.mean((X - v_recon) ** 2))

    def pseudo_likelihood(self, X: np.ndarray) -> float:
        """Stochastic pseudo-log-likelihood (Bengio & Delalleau 2009)."""
        N, D = X.shape
        flip = np.random.randint(0, D, size=N)
        X_c = X.copy()
        if self.spins:
            X_c[np.arange(N), flip] *= -1.0                             # flip ±1
        else:
            X_c[np.arange(N), flip] = 1.0 - X_c[np.arange(N), flip]    # flip {0,1}
        fe = self._free_energy_batch(X)
        fe_c = self._free_energy_batch(X_c)
        return float(np.mean(D * np.logaddexp(0, -(fe_c - fe))))
        
    def annealed_importance_sampling(
        self, 
        n_runs: int = 100, 
        n_betas: int = 1000
    ) -> float:
        """
        Estimates the log partition function ln(Z) of the target RBM using 
        Annealed Importance Sampling (AIS), as recommended in Hinton (2010) Section 14.
        
        The base RBM has the same visible biases as the target, but W = 0 and h_bias = 0.
        Handles Bernoulli, spins (±1) and potts hidden/visible units correctly.
        """
        v_bias_base = self.v_bias.copy()
        
        # --- 1. Exact log-partition of the base RBM (W=0, h_bias=0) --------
        # Visible contribution
        if self.spins:
            # ±1 visible: Π_i (e^{a_i} + e^{-a_i}) = Π_i 2·cosh(a_i)
            abs_a = np.abs(v_bias_base)
            ln_Z_vis = np.sum(abs_a + np.log1p(np.exp(-2.0 * abs_a)))
        else:
            # Bernoulli: Π_i (1 + e^{a_i})
            ln_Z_vis = np.sum(np.log1p(np.exp(np.clip(v_bias_base, -80, 80))))
        # Hidden contribution (h_bias=0)
        if self.potts:
            # One-hot: Σ_k exp(0) = n_h
            ln_Z_hid = np.log(self.n_h)
        else:
            # Bernoulli or ±1: each unit → 2 states → 2^n_h
            ln_Z_hid = self.n_h * np.log(2.0)
        ln_Z_base = ln_Z_vis + ln_Z_hid
        
        # --- 2. Inverse-temperature schedule --------------------------------
        betas = np.linspace(0.0, 1.0, n_betas)
        ln_w = np.zeros(n_runs)
        
        # --- 3. Sample initial v from the base RBM -------------------------
        if self.spins:
            prob_01 = (np.tanh(np.clip(v_bias_base, -80, 80)) + 1.0) / 2.0
            v = 2.0 * (prob_01 > np.random.rand(n_runs, self.n_v)).astype(float) - 1.0
        else:
            p_v_base = self._sigmoid(v_bias_base)
            v = (p_v_base > np.random.rand(n_runs, self.n_v)).astype(float)
        
        # Helper: unnormalized log-probability at temperature beta
        def _ln_p_star(v_state, beta):
            v_term = v_state @ self.v_bias
            act_h = beta * (v_state @ self.W + self.h_bias)
            if self.potts:
                max_a = act_h.max(axis=1, keepdims=True)
                h_term = max_a.squeeze(axis=1) + np.log(
                    np.sum(np.exp(act_h - max_a), axis=1))
            elif self.spins:
                abs_a = np.abs(act_h)
                h_term = np.sum(abs_a + np.log1p(np.exp(-2.0 * abs_a)),
                                axis=1)
            else:
                h_term = np.sum(np.log1p(np.exp(np.clip(act_h, -80, 80))),
                                axis=1)
            return v_term + h_term

        # --- 4. Annealing loop ----------------------------------------------
        for i in range(n_betas - 1):
            b_current = betas[i]
            b_next = betas[i + 1]
            
            ln_w += _ln_p_star(v, b_next) - _ln_p_star(v, b_current)
            
            # Gibbs step at temperature b_next
            # --- sample hidden ---
            act_h = b_next * (v @ self.W + self.h_bias)
            if self.potts:
                act_stable = act_h - act_h.max(axis=1, keepdims=True)
                e = np.exp(act_stable)
                p_h = e / e.sum(axis=1, keepdims=True)
                h = np.zeros_like(p_h)
                for r in range(n_runs):
                    h[r, np.random.choice(self.n_h, p=p_h[r])] = 1.0
            elif self.spins:
                prob_spin = np.tanh(np.clip(act_h, -80, 80))
                p01 = (prob_spin + 1.0) / 2.0
                h = 2.0 * (p01 > np.random.rand(n_runs, self.n_h)).astype(float) - 1.0
            else:
                p_h = 1.0 / (1.0 + np.exp(-np.clip(act_h, -80, 80)))
                h = (p_h > np.random.rand(n_runs, self.n_h)).astype(float)
            
            # --- sample visible ---
            act_v = h @ (b_next * self.W).T + self.v_bias
            if self.spins:
                prob_spin = np.tanh(np.clip(act_v, -80, 80))
                p01 = (prob_spin + 1.0) / 2.0
                v = 2.0 * (p01 > np.random.rand(n_runs, self.n_v)).astype(float) - 1.0
            else:
                p_v = 1.0 / (1.0 + np.exp(-np.clip(act_v, -80, 80)))
                v = (p_v > np.random.rand(n_runs, self.n_v)).astype(float)
            
        # --- 5. Final estimate of ln(Z_target) -----------------------------
        max_ln_w = np.max(ln_w)
        ln_w_mean = max_ln_w + np.log(np.mean(np.exp(ln_w - max_ln_w)))
        
        ln_Z_target = ln_w_mean + ln_Z_base
        return float(ln_Z_target)

    def log_likelihood_ais(
        self, 
        X: np.ndarray, 
        ln_Z: float
    ) -> float:
        """
        Computes the exact average Log-Likelihood of the dataset X given 
        the estimated log partition function ln_Z from AIS.
        """
        fe = self._free_energy_batch(X)
        return float(np.mean(-fe - ln_Z))
    # ─────────────────────────────────────────────────────────────────────
    #  Single-batch CD update
    # ─────────────────────────────────────────────────────────────────────

    def update(
        self,
        v_batch: np.ndarray,
        *,
        n_cd: int = 1,
        persistent: bool = False,
        lr: float | None = None,
        momentum: float | None = None,
    ) -> dict:
        """
        One CD-k gradient step on a single mini-batch.

        Optimized and corrected according to Hinton (2010):
          · Uses pos_h_state (binary) to start CD reconstruction chain.
          · Uses stochastically sampled binary states for intermediate Gibbs steps.
          · Uses continuous probabilities for final gradient updates (noise reduction).
        """
        lr = lr if lr is not None else self.lr
        mom = momentum if momentum is not None else self.momentum_final
        bs = v_batch.shape[0]

        # --- positive phase ---------------------------------------------------
        pos_h_prob, pos_h_state = self.sample_h(v_batch)

        # --- negative phase (CD-k / PCD) --------------------------------------
        if persistent:
            if self._persistent_v is None:
                self._persistent_v = v_batch.copy()
            v_step_state = self._persistent_v
            _, h_state = self.sample_h(v_step_state)
        else:
            h_state = pos_h_state

        v_prob, v_state = self.sample_v(h_state)

        for _ in range(n_cd - 1):
            _, h_state = self.sample_h(v_state)
            v_prob, v_state = self.sample_v(h_state)

        v_final_for_gradient = v_prob

        if persistent:
            self._persistent_v = v_state

        neg_h_prob, _ = self.sample_h(v_final_for_gradient)

        # --- gradient components -----------------------------------------------
        dW_pos = v_batch.T @ pos_h_prob / bs
        bs_neg = v_final_for_gradient.shape[0]  # may differ from bs in PCD
        dW_neg = v_final_for_gradient.T @ neg_h_prob / bs_neg
        dvb_pos = v_batch.mean(axis=0)
        dvb_neg = v_final_for_gradient.mean(axis=0)
        dhb_pos = pos_h_prob.mean(axis=0)
        dhb_neg = neg_h_prob.mean(axis=0)

        dW = dW_pos - dW_neg
        dvb = dvb_pos - dvb_neg
        dhb = dhb_pos - dhb_neg

        # L2 weight decay + L1 penalty su W
        dW -= self.wd * self.W + self.gamma * np.sign(self.W)

        # --- optimiser step ----------------------------------------------------
        step_norm = 0.0

        if self.optimizer_type == "sgd":
            self._vel_W = mom * self._vel_W + lr * dW
            self._vel_vb = mom * self._vel_vb + lr * dvb
            self._vel_hb = mom * self._vel_hb + lr * dhb
            self.W += self._vel_W
            self.v_bias += self._vel_vb
            self.h_bias += self._vel_hb
            step_norm = np.linalg.norm(self._vel_W)

        elif self.optimizer_type == "rmsprop":
            for p, g, s in zip(
                [self.W, self.v_bias, self.h_bias],
                [dW, dvb, dhb],
                [self._sq_W, self._sq_vb, self._sq_hb],
            ):
                s[:] = self.rho * s + (1 - self.rho) * g ** 2
                step = lr * g / (np.sqrt(s) + self.eps)
                p += step
                if p is self.W:
                    step_norm = np.linalg.norm(step)

        v_recon_prob, _ = self.sample_v(pos_h_state)
        mse_value = float(np.mean((v_batch - v_recon_prob) ** 2))

        return {
            "mse": mse_value,
            "update_ratio": float(step_norm / (np.linalg.norm(self.W) + 1e-10)),
            "sparsity": float(np.mean(pos_h_prob)),
            "weights_max": float(np.max(np.abs(self.W))),
            "dW": dW, "dW_pos": dW_pos, "dW_neg": dW_neg,
            "dvb": dvb, "dvb_pos": dvb_pos, "dvb_neg": dvb_neg,
            "dhb": dhb, "dhb_pos": dhb_pos, "dhb_neg": dhb_neg,
            "v_neg": v_final_for_gradient,
        }

    # ─────────────────────────────────────────────────────────────────────
    #  LR / momentum schedule helpers
    # ─────────────────────────────────────────────────────────────────────

    def _get_lr(self, epoch: int, Nepoch: int, lr_fin: float) -> float:
        if self.lr_schedule == "hinton":
            return 0.1 if epoch <= self.lr_ramp else self.lr
        if self.lr_schedule == "linear":
            q = (epoch - 1) / max(Nepoch - 1, 1)
            return self.lr + (lr_fin - self.lr) * q
        return self.lr  # 'constant'

    def _get_momentum(self, epoch: int) -> float:
        return self.momentum_initial if epoch <= self.momentum_ramp else self.momentum_final

    # ─────────────────────────────────────────────────────────────────────
    #  Full training loop
    # ─────────────────────────────────────────────────────────────────────

    def train(
        self,
        data: np.ndarray,
        *,
        Nepoch: int = 150,
        Nmini: int = 20,
        N_ini: int = 10,
        N_fin: int = 500,
        Nt: int = 2,
        persistent: bool = False,
        lr_fin: float | None = None,
        hinton_init: bool = True,
        history_every: int = 1,
        metrics_every: int = 10,
        n_metrics_samples: int = 300,
        save_dir: str | None = None,
        verbose: bool = True,
    ) -> dict:
        """
        Train with Contrastive Divergence.
        """
        Nd = len(data)
        lr_fin = lr_fin if lr_fin is not None else self.lr

        # --- Hinton visible-bias init -----------------------------------------
        if hinton_init:
            if self.spins:
                # ±1 units: E[v] = tanh(a) → a = arctanh(E[v])
                m = np.clip(data.mean(axis=0), -0.998, 0.998)
                self.v_bias = np.arctanh(m)
            else:
                # {0,1} units: P(v=1) = σ(a) → a = logit(p)
                p = np.clip(data.mean(axis=0), 0.001, 0.999)
                self.v_bias = logit(p)

        # --- Reset PCD chain --------------------------------------------------
        self._persistent_v = None

        # --- Metric sub-sample (fixed throughout training) --------------------
        met_idx = np.random.choice(Nd, min(n_metrics_samples, Nd), replace=False)
        X_met = data[met_idx]

        # --- Accumulators -----------------------------------------------------
        h_ep: list[int] = []
        h_w:  list[np.ndarray] = []
        h_a:  list[np.ndarray] = []
        h_b:  list[np.ndarray] = []
        h_gw: list[np.ndarray] = []
        h_ga: list[np.ndarray] = []
        h_gb: list[np.ndarray] = []
        h_gw_d: list[np.ndarray] = []
        h_gw_m: list[np.ndarray] = []
        h_ga_d: list[np.ndarray] = []
        h_ga_m: list[np.ndarray] = []
        h_mse:  list[float] = []
        h_ratio: list[float] = []
        h_sparsity: list[float] = []
        h_wmax: list[float] = []

        m_ep:  list[int] = []
        m_fed: list[float] = []
        m_fef: list[float] = []
        m_gap: list[float] = []
        m_re:  list[float] = []
        m_pl:  list[float] = []

        indices = np.arange(Nd, dtype=int)

        if verbose:
            tag = f"n_v={self.n_v}  n_h={self.n_h}  Nt={Nt}  opt={self.optimizer_type}"
            if persistent:
                tag += "  PCD"
            print(f"\nTraining RBM  {tag}")
            print("=" * 70)

        # ═════════════════════════════════════════════════════════════════
        for epoch in range(1, Nepoch + 1):
            q = (epoch - 1) / max(Nepoch - 1, 1)
            N = int(N_ini + (N_fin - N_ini) * q ** 2)
            lr = self._get_lr(epoch, Nepoch, lr_fin)
            mom = self._get_momentum(epoch)

            stats = None

            for _ in range(Nmini):
                batch = data[np.random.choice(indices, N, replace=False)]
                stats = self.update(
                    batch, n_cd=Nt, persistent=persistent,
                    lr=lr, momentum=mom,
                )

            # --- history snapshot (last mini-batch of this epoch) ----------
            is_last = epoch == Nepoch
            if history_every > 0 and (epoch % history_every == 0 or is_last):
                h_ep.append(epoch)
                h_w.append(self.W.copy())
                h_a.append(self.v_bias.copy())
                h_b.append(self.h_bias.copy())
                h_gw.append(stats["dW"].copy())
                h_ga.append(stats["dvb"].copy())
                h_gb.append(stats["dhb"].copy())
                h_gw_d.append(stats["dW_pos"].copy())
                h_gw_m.append(stats["dW_neg"].copy())
                h_ga_d.append(stats["dvb_pos"].copy())
                h_ga_m.append(stats["dvb_neg"].copy())
                h_mse.append(stats["mse"])
                h_ratio.append(stats["update_ratio"])
                h_sparsity.append(stats["sparsity"])
                h_wmax.append(stats["weights_max"])

            # --- metrics --------------------------------------------------
            if metrics_every > 0 and (epoch % metrics_every == 0 or is_last):
                X_fan = stats["v_neg"]
                fe_d = self.free_energy(X_met)
                fe_f = self.free_energy(X_fan)
                re = self.reconstruction_error(X_met)
                pl = self.pseudo_likelihood(X_met)

                m_ep.append(epoch)
                m_fed.append(fe_d)
                m_fef.append(fe_f)
                m_gap.append(fe_d - fe_f)
                m_re.append(re)
                m_pl.append(pl)

            if verbose and (epoch % 10 == 0 or is_last):
                g_rms = np.std(stats["dW"])
                extra = f"  RE={m_re[-1]:.4f}" if m_re else ""
                print(
                    f"  epoch {epoch:3d}/{Nepoch}  N={N:3d}  "
                    f"lr={lr:.4f}  mom={mom:.2f}  "
                    f"∇rms={g_rms:.5f}{extra}"
                )
        # ═════════════════════════════════════════════════════════════════

        if verbose:
            print("=" * 70)
            print("Done.\n")

        # --- build results dict ----------------
        results: dict = {
            "config": {
                "D": self.n_v, "L": self.n_h, "seed": self.seed,
                "Nepoch": Nepoch, "Nmini": Nmini,
                "N_ini": N_ini, "N_fin": N_fin, "Nt": Nt,
                "optimizer": self.optimizer_type,
                "lr": self.lr, "lr_schedule": self.lr_schedule,
                "lr_fin": lr_fin, "lr_ramp": self.lr_ramp,
                "momentum": self.momentum_final,
                "momentum_initial": self.momentum_initial,
                "momentum_ramp": self.momentum_ramp,
                "weight_decay": self.wd, "gamma": self.gamma,
                "rmsprop_rho": self.rho, "rmsprop_eps": self.eps,
                "spins": self.spins, "potts": self.potts,
                "persistent": persistent, "hinton_init": hinton_init,
                "history_every": history_every,
                "metrics_every": metrics_every,
                "n_metrics_samples": n_metrics_samples, "Nd": Nd,
            },
            "weights": {
                "w": self.W.copy(),
                "a": self.v_bias.copy(),
                "b": self.h_bias.copy(),
            },
            "history": {
                "epochs":   np.array(h_ep),
                "w":        np.array(h_w)   if h_w else np.empty(0),
                "a":        np.array(h_a)   if h_a else np.empty(0),
                "b":        np.array(h_b)   if h_b else np.empty(0),
                "gw":       np.array(h_gw)  if h_gw else np.empty(0),
                "ga":       np.array(h_ga)  if h_ga else np.empty(0),
                "gb":       np.array(h_gb)  if h_gb else np.empty(0),
                "gw_data":  np.array(h_gw_d) if h_gw_d else np.empty(0),
                "gw_model": np.array(h_gw_m) if h_gw_m else np.empty(0),
                "ga_data":  np.array(h_ga_d) if h_ga_d else np.empty(0),
                "ga_model": np.array(h_ga_m) if h_ga_m else np.empty(0),
                "mse":      np.array(h_mse),
                "update_ratio": np.array(h_ratio),
                "sparsity": np.array(h_sparsity),
                "weights_max": np.array(h_wmax),
            },
            "metrics": {
                "epochs":              np.array(m_ep),
                "free_energy_data":    np.array(m_fed),
                "free_energy_fantasy": np.array(m_fef),
                "free_energy_gap":     np.array(m_gap),
                "reconstruction_error": np.array(m_re),
                "pseudo_likelihood":   np.array(m_pl),
            },
        }

        if save_dir is not None:
            save_results(results, save_dir)

        return results

    # ─────────────────────────────────────────────────────────────────────
    #  Utilities
    # ─────────────────────────────────────────────────────────────────────

    def sample_chain(self, x_init: np.ndarray, n_steps: int = 100) -> np.ndarray:
        """
        Run a Gibbs chain for *n_steps*.
        """
        squeeze = x_init.ndim == 1
        x = x_init[np.newaxis, :].copy() if squeeze else x_init.copy()
        shape = (n_steps,) + x.shape
        chain = np.empty(shape)

        for t in range(n_steps):
            _, h_state = self.sample_h(x)
            v_prob, _ = self.sample_v(h_state)
            if self.spins:
                p01 = (v_prob + 1.0) / 2.0
                x = 2.0 * (p01 > np.random.rand(*p01.shape)).astype(float) - 1.0
            else:
                x = (v_prob > np.random.rand(*v_prob.shape)).astype(float)
            chain[t] = x

        return chain[:, 0, :] if squeeze else chain

    def energy(self, x: np.ndarray, z: np.ndarray) -> float:
        """Joint energy  E(x, z) = −[v_bias·x + h_bias·z + x^T W z]."""
        return float(-(self.v_bias @ x + self.h_bias @ z + x @ self.W @ z))

    def load_weights(self, results: dict):
        """Restore model weights from a results dict."""
        self.W = results["weights"]["w"].copy()
        self.v_bias = results["weights"]["a"].copy()
        self.h_bias = results["weights"]["b"].copy()


# ═════════════════════════════════════════════════════════════════════════
#  PERSISTENCE
# ═════════════════════════════════════════════════════════════════════════

def save_results(results: dict, path: str | Path) -> Path:
    """Save results to *path/* (config.json + weights/history/metrics.npz)."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

    with open(p / "config.json", "w") as f:
        json.dump(results["config"], f, indent=2)

    np.savez_compressed(p / "weights.npz", **results["weights"])
    np.savez_compressed(p / "history.npz", **results["history"])
    np.savez_compressed(p / "metrics.npz", **results["metrics"])
    return p


def load_results(path: str | Path) -> dict:
    """Load results saved by :func:`save_results`."""
    p = Path(path)
    with open(p / "config.json") as f:
        config = json.load(f)

    def _load(name):
        with np.load(p / name) as npz:
            return {k: npz[k] for k in npz.files}

    return {
        "config": config,
        "weights": _load("weights.npz"),
        "history": _load("history.npz"),
        "metrics": _load("metrics.npz"),
    }


# ═════════════════════════════════════════════════════════════════════════
#  STANDALONE HELPERS
# ═════════════════════════════════════════════════════════════════════════

def compute_prototypes(data: np.ndarray, labels: np.ndarray) -> dict:
    """Mean image per class → {label: mean_vector}."""
    labels = np.asarray(labels, dtype=int)
    return {int(c): data[labels == c].mean(axis=0) for c in np.unique(labels)}


def classify_nearest(x: np.ndarray, prototypes: dict) -> int:
    """Nearest-prototype classification (L2)."""
    dists = {c: float(np.sum((x - p) ** 2)) for c, p in prototypes.items()}
    return min(dists, key=dists.get)