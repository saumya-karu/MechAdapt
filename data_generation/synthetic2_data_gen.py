#%%

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


class RNG:
    def __init__(self, seed=None):
        self.seed = seed
        try:
            self._rng = np.random.default_rng(seed)
            self._is_new = True
        except AttributeError:
            self._rng = np.random.RandomState(seed)
            self._is_new = False

    def rand(self, *size):
        if self._is_new:
            return self._rng.random(size if size else None)
        return self._rng.rand(*size) if size else self._rng.rand()

    def normal(self, loc=0.0, scale=1.0, size=None):
        return self._rng.normal(loc=loc, scale=scale, size=size)

    def uniform(self, low=0.0, high=1.0, size=None):
        return self._rng.uniform(low, high, size=size)

    def randint(self, low, high=None, size=None):
        if self._is_new:
            return self._rng.integers(low, high=high, size=size)
        return self._rng.randint(low, high=high, size=size)


def get_rng(seed=None) -> RNG:
    return RNG(seed)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def split_indices(T: int, train_frac: float = 0.2, val_frac: float = 0.1) -> Tuple[int, int]:
    train_end = int(T * train_frac)
    val_end = train_end + int(T * val_frac)
    return train_end, val_end


def add_date_column(df: pd.DataFrame, start: str = "1990/1/1 0:00") -> pd.DataFrame:
    T = len(df)
    dt = pd.date_range(start=start, periods=T, freq="H")
    out = df.copy()
    out.insert(0, "date", [f"{d.year}/{d.month}/{d.day} {d.hour}:00" for d in dt])
    return out



def get_regime_id(T: int, kind: str) -> np.ndarray:
    if kind != "abrupt":
        return np.zeros(T, dtype=int)
    t = np.arange(T)
    reg = np.zeros(T, dtype=int)
    reg[t >= T // 3] = 1
    reg[t >= 2 * T // 3] = 2
    return reg


def get_weight_matrix(T: int, kind: str = "abrupt", n_cycles_periodic: int = 3) -> np.ndarray:
    t = np.arange(T)
    W = np.zeros((T, 3), dtype=float)

    if kind == "abrupt":
        reg = get_regime_id(T, "abrupt")
        W[reg == 0] = np.array([1.0, 0.25, 0.15])
        W[reg == 1] = np.array([0.25, 0.9, 0.35])
        W[reg == 2] = np.array([-0.6, 0.15, 1.2])

    elif kind == "periodic":
        t_norm = t / max(T - 1, 1)
        W[:, 0] = 0.7 * np.sin(2 * np.pi * n_cycles_periodic * t_norm) + 0.2
        W[:, 1] = 0.6 + 0.25 * np.cos(2 * np.pi * n_cycles_periodic * t_norm)
        W[:, 2] = 0.8 - 0.35 * np.sin(2 * np.pi * n_cycles_periodic * t_norm + np.pi / 3) + 0.15
    else:
        W[:] = np.array([0.6, 0.4, 0.2])

    return W

def gen_rich_features_stationary(T: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = get_rng(seed)
    t = np.arange(T)
    t_norm = t / max(T - 1, 1)

    s_fast = np.sin(2 * np.pi * t / 48)
    s_slow = np.sin(2 * np.pi * t / 336)
    s_mid  = np.sin(2 * np.pi * t / 96)

    rw = np.cumsum(rng.normal(0, 0.03, size=T))

    ar = np.zeros(T)
    rho = 0.8
    for i in range(1, T):
        ar[i] = rho * ar[i - 1] + rng.normal(0, 0.1)

    X1 = 0.7 * ar + 0.55 * s_fast + 0.25 * rw + rng.normal(0, 0.06, size=T)
    X2 = 0.35 * ar + 0.9 * s_slow + 0.55 * rw + 1.8 * t_norm + (t ** 0.5) * 0.05
    X3 = 0.6 * ar + 0.35 * rw + 0.6 * np.sin(2 * np.pi * t / 72 + np.pi / 6) + 0.35 * np.sin(2 * np.pi * t / 18) + 0.2 * s_mid

    def standardise(x):
        x = x - x.mean()
        return x / (x.std() + 1e-6)

    return standardise(X1), standardise(X2), standardise(X3)


def lag_arr(x: np.ndarray, k: int) -> np.ndarray:

    k = int(k)
    if k <= 0:
        return x.copy()
    out = np.zeros_like(x)
    out[k:] = x[:-k]
    return out



def gen_causal_lagged_dataset(
    T: int = 10000,
    seed: int = 1,
    noise_std: float = 0.15,
    kind: str = "abrupt",
    y_scale: float = 5.0,
    n_cycles_periodic: int = 3,

    #ground-truth lags
    lag_X1: int = 0,
    lag_X2: int = 3,
    lag_X3: int = 7,
):
    rng = get_rng(seed)

    X1, X2, X3 = gen_rich_features_stationary(T, seed=seed)

    t = np.arange(T)
    t_norm = t / max(T - 1, 1)

    W = get_weight_matrix(T, kind=kind, n_cycles_periodic=n_cycles_periodic)
    regime_id = get_regime_id(T, kind=kind)

    X1_l = lag_arr(X1, lag_X1)
    X2_l = lag_arr(X2, lag_X2)
    X3_l = lag_arr(X3, lag_X3)


    phi1 = np.tanh(X1_l) * (1 + 0.25 * np.sin(2 * np.pi * t_norm))
    phi2 = (X2_l ** 2) * (0.9 + 0.1 * np.cos(2 * np.pi * t_norm))
    phi3 = np.sin(X3_l) * (1 + 0.4 * t_norm)

    core = (W * np.stack([phi1, phi2, phi3], axis=1)).sum(axis=1)
    core_z = (core - core.mean()) / (core.std() + 1e-6)

    eps = rng.normal(0, noise_std, size=T)
    Y = y_scale * core_z + eps

    df = pd.DataFrame({
        "X1": X1,
        "X2": X2,
        "X3": X3,
        "Y": Y,
        "regime_id": regime_id,
    })


    df["lag_X1"] = int(lag_X1)
    df["lag_X2"] = int(lag_X2)
    df["lag_X3"] = int(lag_X3)

    return df, W


def sample_drift_blocks_in_test_width(
    T: int,
    val_end: int,
    n_blocks: int,
    drift_width: int,
    seed: int = 0,
) -> List[Tuple[int, int]]:
    rng = get_rng(seed)

    test_start = int(val_end)
    test_end = int(T)
    if test_start >= test_end:
        return []

    drift_width = int(max(1, min(int(drift_width), test_end - test_start)))
    start_min = test_start
    start_max = test_end - drift_width
    if start_max < start_min:
        return []

    blocks: List[Tuple[int, int]] = []
    attempts = 0
    max_attempts = max(400, n_blocks * 400)

    while len(blocks) < n_blocks and attempts < max_attempts:
        attempts += 1
        s = int(rng.randint(start_min, start_max + 1))
        e = s + drift_width
        if any(not (e <= bs or s >= be) for (bs, be) in blocks):
            continue
        blocks.append((s, e))

    return sorted(blocks)


def apply_spurious_drift_block_xonly(
    df: pd.DataFrame,
    df_orig: pd.DataFrame,
    cols: List[str],
    start: int,
    end: int,
    rng: RNG,
    scale_range: Tuple[float, float] = (0.5, 3.0),
    shift_std_mult: float = 3.0,
    block_id: int = 1,
) -> Dict:
    drift_params = []
    direction = 1 if (block_id % 2 == 1) else -1

    for col in cols:
        if col not in df.columns:
            continue
        if col in ("Y", "regime_id", "env_id", "date"):
            continue

        col_std = float(df_orig[col].std())
        if not np.isfinite(col_std) or col_std <= 0:
            col_std = 1.0

        #scale
        if direction == 1:
            s = float(rng.uniform(max(1.2, scale_range[0]), max(1.6, scale_range[1])))
        else:
            low = min(scale_range[0], 0.3)
            high = min(scale_range[1], 0.8)
            s = float(rng.uniform(low, high))

        shift
        shift = float(direction * rng.normal(loc=0.0, scale=shift_std_mult * col_std))

        base_vals = df_orig.loc[df_orig.index[start:end], col].to_numpy()
        df.loc[df.index[start:end], col] = s * base_vals + shift

        drift_params.append({"col": col, "scale": s, "shift": shift})

    return {
        "block_id": int(block_id),
        "start_idx": int(start),
        "end_idx": int(end),
        "affected_cols": [p["col"] for p in drift_params],
        "params": drift_params,
    }


def inject_spurious_drifts_test_only(
    df: pd.DataFrame,
    n_blocks: int = 10,
    drift_width: int = 200,
    seed_spurious: int = 123,
    scale_range: Tuple[float, float] = (0.3, 4.0),
    shift_std_mult: float = 10.0,
    drift_prob: float = 1.0,
    drift_cols: Tuple[str, ...] = ("X1", "X2", "X3"),
    train_frac: float = 0.2,
    val_frac: float = 0.1,
) -> Tuple[pd.DataFrame, List[Dict], List[Tuple[int, int]]]:
    rng = get_rng(seed_spurious)

    out = df.copy()
    df_orig = df.copy()
    T = len(out)

    train_end, val_end = split_indices(T, train_frac=train_frac, val_frac=val_frac)

    drift_cols_list = [c for c in drift_cols if c in out.columns and c not in ("Y", "regime_id")]
    blocks = sample_drift_blocks_in_test_width(
        T=T, val_end=val_end, n_blocks=n_blocks, drift_width=drift_width, seed=seed_spurious
    )

    env_id = np.zeros(T, dtype=int)
    meta: List[Dict] = []

    for block_id, (start, end) in enumerate(blocks, start=1):
        if rng.rand() > drift_prob:
            continue
        block_meta = apply_spurious_drift_block_xonly(
            df=out,
            df_orig=df_orig,
            cols=drift_cols_list,
            start=start,
            end=end,
            rng=rng,
            scale_range=scale_range,
            shift_std_mult=shift_std_mult,
            block_id=block_id,
        )
        env_id[start:end] = block_id
        meta.append(block_meta)

    out["env_id"] = env_id


    if not np.allclose(out["Y"].to_numpy(), df_orig["Y"].to_numpy()):
        raise AssertionError("Y changed during spurious drift injection (should never happen).")

    return out, meta, blocks


@dataclass
class Synthetic2Config:
    T: int = 10000
    seed_dataset: int = 2
    kind: str = "abrupt"
    noise_std: float = 0.15
    y_scale: float = 5.0

    lag_X1: int = 0
    lag_X2: int = 3
    lag_X3: int = 7

    add_spurious: bool = True
    seed_spurious: int = 123
    n_blocks: int = 20
    drift_width: int = 200
    scale_range: Tuple[float, float] = (0.3, 4.0)
    shift_std_mult: float = 10.0
    drift_prob: float = 1.0
    drift_cols: Tuple[str, ...] = ("X1", "X2", "X3")
    train_frac: float = 0.2
    val_frac: float = 0.1

    start_date: str = "1990/1/1 0:00"
    data_dir: str = r""
    output_dir: str = r""


def make_and_save_synthetic2(cfg: Synthetic2Config):
    set_seed(cfg.seed_dataset)

    df_clean, W = gen_causal_lagged_dataset(
        T=cfg.T,
        seed=cfg.seed_dataset,
        noise_std=cfg.noise_std,
        kind=cfg.kind,
        y_scale=cfg.y_scale,
        lag_X1=cfg.lag_X1,
        lag_X2=cfg.lag_X2,
        lag_X3=cfg.lag_X3,
    )
    df_clean["env_id"] = 0

    meta, blocks = [], []
    if cfg.add_spurious:
        df_spur, meta, blocks = inject_spurious_drifts_test_only(
            df=df_clean.drop(columns=["env_id"]),
            n_blocks=cfg.n_blocks,
            drift_width=cfg.drift_width,
            seed_spurious=cfg.seed_spurious,
            scale_range=cfg.scale_range,
            shift_std_mult=cfg.shift_std_mult,
            drift_prob=cfg.drift_prob,
            drift_cols=cfg.drift_cols,
            train_frac=cfg.train_frac,
            val_frac=cfg.val_frac,
        )
    else:
        df_spur = df_clean.copy()
        df_spur["env_id"] = 0

    df_clean_date = add_date_column(df_clean, start=cfg.start_date)
    df_spur_date = add_date_column(df_spur, start=cfg.start_date)

    keep_cols = ["date", "X1", "X2", "X3", "Y"]
    df_clean_model = df_clean_date[keep_cols].copy()
    df_spur_model  = df_spur_date[keep_cols].copy()

    DATA_DIR = Path(cfg.data_dir); DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR = Path(cfg.output_dir); OUT_DIR.mkdir(parents=True, exist_ok=True)

    clean_path = DATA_DIR / "synthetic2.csv"
    df_clean_model.to_csv(clean_path, index=False)

    spur_name = f"synthetic2_{cfg.drift_width}_{cfg.n_blocks}.csv"
    spur_path = OUT_DIR / spur_name
    df_spur_model.to_csv(spur_path, index=False)

    print("Saved CLEAN   :", clean_path)
    print("Saved SPURIOUS:", spur_path)
    return df_clean_model, df_spur_model, W, meta, blocks, clean_path, spur_path


if __name__ == "__main__":
    cfg = Synthetic2Config(
        T=10000,
        seed_dataset=2,
        kind="abrupt",
        noise_std=0.15,
        y_scale=5.0,
        lag_X1=0,
        lag_X2=3,
        lag_X3=7,
        add_spurious=True,
        seed_spurious=123,
        n_blocks=30,
        drift_width=200,
        drift_prob=1.0,
        shift_std_mult=10.0,
        scale_range=(0.3, 4.0),
        drift_cols=("X1","X2","X3"),
    )
    make_and_save_synthetic2(cfg)


#%%
