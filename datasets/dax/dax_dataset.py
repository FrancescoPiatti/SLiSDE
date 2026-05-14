# experiments/datasets/dax_dataset.py
"""
Load and preprocess DAX options data for calibration experiments.

Reads a CSV file with columns:
    [Instrument, PUTCALLIND, EXPIR_DATE, STRIKE_PRC, CF_CLOSE, IMP_VOLT]

Normalises everything by the DAX spot price (S0 = 25280):
    K_norm = K / S0,   price_norm = market_price / S0,   Y_0 = 1.0.

Produces a DaxOptionsDataset containing:
    - Vanilla call/put targets (ATM band 0.90--1.10)
    - OTM put targets (barrier proxy, 0.75--0.92)
    - Maturity grid and index mapping
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# Dataset dataclass
# ─────────────────────────────────────────────────────────────────────

@dataclass
class DaxOptionsDataset:
    """Pre-processed DAX options data for model calibration.

    All strikes and prices are normalised by the spot price S0.

    Attributes:
        call_strikes:           (n_call_strikes,) normalised call strike grid.
        call_prices:            (n_call_strikes, n_maturities) normalised call prices.
        put_strikes:            (n_put_strikes,) normalised put strike grid.
        put_prices:             (n_put_strikes, n_maturities) normalised put prices.
        otm_put_strikes:        (n_otm_strikes,) normalised OTM put strike grid.
        otm_put_prices:         (n_otm_strikes, n_maturities) normalised OTM put prices.
        otm_call_strikes:       (n_otm_call_strikes,) normalised deep OTM call strike grid.
        otm_call_prices:        (n_otm_call_strikes, n_maturities) normalised deep OTM call prices.
        digital_put_strikes:    (n_digital_strikes,) normalised digital put strike grid.
        digital_put_prices:     (n_digital_strikes, n_maturities) digital put prices from put spreads.
        maturities:             (n_maturities,) time-to-maturity in years.
        maturity_indices:       (n_maturities,) integer indices into model time grid.
        spot:                   DAX spot level.
        y0:                     (1,) normalised starting value [1.0].
        T:                      model terminal time (1.0).
        raw_df:                 full filtered DataFrame for plotting / inspection.
    """
    call_strikes: jnp.ndarray
    call_prices: jnp.ndarray
    put_strikes: jnp.ndarray
    put_prices: jnp.ndarray
    otm_put_strikes: jnp.ndarray
    otm_put_prices: jnp.ndarray
    otm_call_strikes: jnp.ndarray
    otm_call_prices: jnp.ndarray
    digital_put_strikes: jnp.ndarray
    digital_put_prices: jnp.ndarray
    maturities: jnp.ndarray
    maturity_indices: jnp.ndarray
    spot: float
    y0: jnp.ndarray
    T: float
    raw_df: pd.DataFrame = field(default=None, repr=False)


# ─────────────────────────────────────────────────────────────────────
# Helper: subsample strikes via nearest-neighbour interpolation
# ─────────────────────────────────────────────────────────────────────

def _subsample_strikes(
    df_slice: pd.DataFrame,
    k_lo: float,
    k_hi: float,
    n_strikes: int,
    spot: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select n_strikes evenly spaced normalised strikes in [k_lo, k_hi].

    For each target strike, picks the nearest available strike from the
    data and reads its normalised price.

    Args:
        df_slice:  DataFrame slice for one maturity+type.
        k_lo:      lower bound for normalised strike K/S0.
        k_hi:      upper bound for normalised strike K/S0.
        n_strikes: number of strikes to select.
        spot:      spot price for normalisation.

    Returns:
        selected_strikes: (n_strikes,) normalised strikes actually used.
        selected_prices:  (n_strikes,) normalised prices.
    """
    if df_slice.empty:
        return np.full(n_strikes, np.nan), np.full(n_strikes, np.nan)

    available_k = df_slice["STRIKE_PRC"].values / spot
    available_p = df_slice["CF_CLOSE"].values / spot

    # Sort by strike
    order = np.argsort(available_k)
    available_k = available_k[order]
    available_p = available_p[order]

    # Filter to range
    mask = (available_k >= k_lo) & (available_k <= k_hi)
    if mask.sum() < 2:
        return np.full(n_strikes, np.nan), np.full(n_strikes, np.nan)

    avail_k_in = available_k[mask]
    avail_p_in = available_p[mask]

    # Target grid
    target_k = np.linspace(k_lo, k_hi, n_strikes)

    # Nearest-neighbour + linear interpolation for prices
    selected_p = np.interp(target_k, avail_k_in, avail_p_in)
    selected_k = target_k

    return selected_k, selected_p


# ─────────────────────────────────────────────────────────────────────
# Main loader
# ─────────────────────────────────────────────────────────────────────

# Default 5 maturities (expiry dates)
DEFAULT_EXPIRIES = [
    "2026-04-17",
    "2026-06-19",
    "2026-09-18",
    "2026-12-18",
    "2027-03-19",
]

# Reference date for TTM computation
DEFAULT_REF_DATE = "2026-03-31"


def load_dax_options(
    csv_path: Optional[str] = None,
    spot: float = 25280.0,
    ref_date: str = DEFAULT_REF_DATE,
    expiry_dates: Optional[List[str]] = None,
    n_call_strikes: int = 10,
    n_put_strikes: int = 10,
    n_otm_strikes: int = 10,
    n_otm_call_strikes: int = 8,
    n_digital_strikes: int = 6,
    call_k_range: Tuple[float, float] = (0.90, 1.10),
    put_k_range: Tuple[float, float] = (0.90, 1.10),
    otm_k_range: Tuple[float, float] = (0.60, 0.90),
    otm_call_k_range: Tuple[float, float] = (1.10, 1.35),
    digital_k_range: Tuple[float, float] = (0.70, 0.95),
    digital_spread_width: float = 0.02,
    model_T: float = 1.0,
    model_n_steps: int = 256,
) -> DaxOptionsDataset:
    """Load DAX options CSV and build a calibration dataset.

    Args:
        csv_path:            path to dax_options.csv.  If None, auto-detected.
        spot:                DAX spot price (default 25280).
        ref_date:            reference date for TTM computation.
        expiry_dates:        list of expiry date strings (YYYY-MM-DD).
        n_call_strikes:      number of call strikes per maturity.
        n_put_strikes:       number of put strikes per maturity.
        n_otm_strikes:       number of OTM put strikes per maturity.
        n_otm_call_strikes:  number of deep OTM call strikes per maturity.
        n_digital_strikes:   number of digital put strikes per maturity.
        call_k_range:        (lo, hi) normalised strike range for calls.
        put_k_range:         (lo, hi) normalised strike range for puts.
        otm_k_range:         (lo, hi) normalised strike range for OTM puts.
        otm_call_k_range:    (lo, hi) normalised strike range for deep OTM calls.
        digital_k_range:     (lo, hi) normalised strike range for digital puts.
        digital_spread_width: width of the put spread used to approximate digitals.
        model_T:             model terminal time.
        model_n_steps:       model time discretisation steps.

    Returns:
        DaxOptionsDataset with all calibration targets.
    """
    if csv_path is None:
        # Search in datasets/dax/ first, then fallback locations
        candidates = [
            os.path.join(os.path.dirname(__file__), "dax_options.csv"),
            os.path.join("datasets", "dax", "dax_options.csv"),
            os.path.join("experiments", "datasets", "dax_options.csv"),
            os.path.join("experiments", "exp_opt", "dax_options.csv"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                csv_path = candidate
                break
        if csv_path is None:
            raise FileNotFoundError(
                f"dax_options.csv not found in any of: {candidates}"
            )

    if expiry_dates is None:
        expiry_dates = DEFAULT_EXPIRIES

    # ── 1. Read and clean CSV ────────────────────────────────────────
    df = pd.read_csv(csv_path)
    df["PUTCALLIND"] = df["PUTCALLIND"].str.strip()
    df["EXPIR_DATE"] = pd.to_datetime(df["EXPIR_DATE"])
    ref = pd.Timestamp(ref_date)

    # Compute TTM in years (ACT/365)
    df["TTM"] = (df["EXPIR_DATE"] - ref).dt.days / 365.0

    # Drop expired options (TTM <= 0)
    df = df[df["TTM"] > 0].copy()

    # Drop IMP_VOLT column if present (not used)
    if "IMP_VOLT" in df.columns:
        df = df.drop(columns=["IMP_VOLT"])

    # Drop rows with missing prices
    df = df.dropna(subset=["CF_CLOSE"])
    df = df[df["CF_CLOSE"] > 0].copy()

    print(f"Loaded {len(df)} option records after filtering (TTM > 0, price > 0)")

    # ── 2. Select 5 maturities ──────────────────────────────────────
    target_expiries = pd.to_datetime(expiry_dates)
    df_selected = df[df["EXPIR_DATE"].isin(target_expiries)].copy()

    if len(df_selected) == 0:
        raise ValueError(
            f"No options found for target expiries {expiry_dates}. "
            f"Available expiries: {sorted(df['EXPIR_DATE'].unique())}"
        )

    actual_expiries = sorted(df_selected["EXPIR_DATE"].unique())
    print(f"Selected {len(actual_expiries)} maturities:")

    maturities_years = []
    for exp in actual_expiries:
        ttm = (exp - ref).days / 365.0
        maturities_years.append(ttm)
        n_calls = len(df_selected[(df_selected["EXPIR_DATE"] == exp) & (df_selected["PUTCALLIND"] == "CALL")])
        n_puts = len(df_selected[(df_selected["EXPIR_DATE"] == exp) & (df_selected["PUTCALLIND"] == "PUT")])
        print(f"  {exp.date()}: TTM={ttm:.3f}y, {n_calls} calls, {n_puts} puts")

    maturities_years = np.array(maturities_years)

    # ── 3. Compute maturity indices into model time grid ─────────────
    # Model runs from t=0 to t=T with n_steps steps
    # maturity_index = round(TTM / T * n_steps)
    mat_indices = np.round(maturities_years / model_T * model_n_steps).astype(int)
    mat_indices = np.clip(mat_indices, 1, model_n_steps)
    print(f"Maturity indices: {mat_indices}")

    # ── 4. Build strike grids and price matrices ─────────────────────
    n_mats = len(actual_expiries)

    # Call strikes: shared grid across maturities
    call_strikes_grid = np.linspace(call_k_range[0], call_k_range[1], n_call_strikes)
    call_prices_mat = np.zeros((n_call_strikes, n_mats))

    put_strikes_grid = np.linspace(put_k_range[0], put_k_range[1], n_put_strikes)
    put_prices_mat = np.zeros((n_put_strikes, n_mats))

    otm_strikes_grid = np.linspace(otm_k_range[0], otm_k_range[1], n_otm_strikes)
    otm_prices_mat = np.zeros((n_otm_strikes, n_mats))

    otm_call_strikes_grid = np.linspace(otm_call_k_range[0], otm_call_k_range[1], n_otm_call_strikes)
    otm_call_prices_mat = np.zeros((n_otm_call_strikes, n_mats))

    # Digital put strikes and prices (approximated via put spreads)
    digital_strikes_grid = np.linspace(digital_k_range[0], digital_k_range[1], n_digital_strikes)
    digital_prices_mat = np.zeros((n_digital_strikes, n_mats))

    for i, exp in enumerate(actual_expiries):
        df_exp = df_selected[df_selected["EXPIR_DATE"] == exp]

        # --- Calls (ATM band) ---
        df_calls = df_exp[df_exp["PUTCALLIND"] == "CALL"]
        if not df_calls.empty:
            k_sel, p_sel = _subsample_strikes(
                df_calls, call_k_range[0], call_k_range[1], n_call_strikes, spot
            )
            call_prices_mat[:, i] = p_sel
        else:
            call_prices_mat[:, i] = np.nan

        # --- Puts (ATM band) ---
        df_puts = df_exp[df_exp["PUTCALLIND"] == "PUT"]
        if not df_puts.empty:
            k_sel, p_sel = _subsample_strikes(
                df_puts, put_k_range[0], put_k_range[1], n_put_strikes, spot
            )
            put_prices_mat[:, i] = p_sel
        else:
            put_prices_mat[:, i] = np.nan

        # --- OTM puts (deep) ---
        if not df_puts.empty:
            k_sel, p_sel = _subsample_strikes(
                df_puts, otm_k_range[0], otm_k_range[1], n_otm_strikes, spot
            )
            otm_prices_mat[:, i] = p_sel
        else:
            otm_prices_mat[:, i] = np.nan

        # --- Deep OTM calls ---
        if not df_calls.empty:
            k_sel, p_sel = _subsample_strikes(
                df_calls, otm_call_k_range[0], otm_call_k_range[1], n_otm_call_strikes, spot
            )
            otm_call_prices_mat[:, i] = p_sel
        else:
            otm_call_prices_mat[:, i] = np.nan

        # --- Digital puts (approximated via put spreads) ---
        # Digital put ≈ [P(K) - P(K - ε)] / ε  where ε = spread_width * S0
        if not df_puts.empty:
            avail_k = df_puts["STRIKE_PRC"].values / spot
            avail_p = df_puts["CF_CLOSE"].values / spot
            order = np.argsort(avail_k)
            avail_k = avail_k[order]
            avail_p = avail_p[order]
            # Interpolate put prices at K and K - ε
            p_at_k = np.interp(digital_strikes_grid, avail_k, avail_p, left=np.nan, right=np.nan)
            p_at_k_minus = np.interp(digital_strikes_grid - digital_spread_width, avail_k, avail_p, left=np.nan, right=np.nan)
            digital_prices_mat[:, i] = (p_at_k - p_at_k_minus) / digital_spread_width
        else:
            digital_prices_mat[:, i] = np.nan

    # Replace any remaining NaN with 0 (strikes outside available data)
    call_prices_mat = np.nan_to_num(call_prices_mat, nan=0.0)
    put_prices_mat = np.nan_to_num(put_prices_mat, nan=0.0)
    otm_prices_mat = np.nan_to_num(otm_prices_mat, nan=0.0)
    otm_call_prices_mat = np.nan_to_num(otm_call_prices_mat, nan=0.0)
    digital_prices_mat = np.clip(np.nan_to_num(digital_prices_mat, nan=0.0), 0.0, 1.0)

    n_vanilla = (n_call_strikes + n_put_strikes) * n_mats
    n_otm = (n_otm_strikes + n_otm_call_strikes) * n_mats
    n_digital = n_digital_strikes * n_mats
    print(f"\nCalibration grid summary:")
    print(f"  Calls: {n_call_strikes} strikes in [{call_k_range[0]:.2f}, {call_k_range[1]:.2f}]")
    print(f"  Puts:  {n_put_strikes} strikes in [{put_k_range[0]:.2f}, {put_k_range[1]:.2f}]")
    print(f"  OTM puts: {n_otm_strikes} strikes in [{otm_k_range[0]:.2f}, {otm_k_range[1]:.2f}]")
    print(f"  OTM calls: {n_otm_call_strikes} strikes in [{otm_call_k_range[0]:.2f}, {otm_call_k_range[1]:.2f}]")
    print(f"  Digital puts: {n_digital_strikes} strikes in [{digital_k_range[0]:.2f}, {digital_k_range[1]:.2f}]")
    print(f"  Maturities: {n_mats} ({maturities_years})")
    print(f"  Total targets: {n_vanilla} vanilla + {n_otm} OTM + {n_digital} digital = {n_vanilla + n_otm + n_digital}")

    # ── 5. Build dataset ─────────────────────────────────────────────
    dataset = DaxOptionsDataset(
        call_strikes=jnp.array(call_strikes_grid),
        call_prices=jnp.array(call_prices_mat),
        put_strikes=jnp.array(put_strikes_grid),
        put_prices=jnp.array(put_prices_mat),
        otm_put_strikes=jnp.array(otm_strikes_grid),
        otm_put_prices=jnp.array(otm_prices_mat),
        otm_call_strikes=jnp.array(otm_call_strikes_grid),
        otm_call_prices=jnp.array(otm_call_prices_mat),
        digital_put_strikes=jnp.array(digital_strikes_grid),
        digital_put_prices=jnp.array(digital_prices_mat),
        maturities=jnp.array(maturities_years),
        maturity_indices=jnp.array(mat_indices),
        spot=spot,
        y0=jnp.array([1.0]),
        T=model_T,
        raw_df=df_selected,
    )

    return dataset
