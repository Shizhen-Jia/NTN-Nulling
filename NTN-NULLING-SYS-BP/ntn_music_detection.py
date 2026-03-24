"""MUSIC-based narrowband detection utilities for NTN interference.

This module is designed for the per-BS channel tensor:
    hi.shape == (num_ntn, num_ntn_ant, num_bs_ant)

Typical usage in your notebook:
    from ntn_music_detection import detect_ntn_music_from_hi
    out = detect_ntn_music_from_hi(hi=h_i, num_sources=None, threshold=3.0)
    mask = ~out["detected_mask_user"]   # keep old semantics if needed
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import numpy as np


def _as_complex_array(x: np.ndarray) -> np.ndarray:
    """Convert input to a complex ndarray without modifying the original."""
    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        return arr.astype(np.complex128, copy=False)
    return arr.astype(np.complex128) + 0j


def _broadcast_powers(
    user_powers: Optional[np.ndarray],
    num_ntn: int,
    num_ntn_ant: int,
) -> np.ndarray:
    """Broadcast user powers to shape (num_ntn, num_ntn_ant)."""
    if user_powers is None:
        return np.ones((num_ntn, num_ntn_ant), dtype=np.float64)

    p = np.asarray(user_powers, dtype=np.float64)
    if p.ndim == 1:
        if p.shape[0] != num_ntn:
            raise ValueError(
                "user_powers has shape (num_ntn,), but num_ntn does not match."
            )
        return np.repeat(p[:, None], num_ntn_ant, axis=1)

    if p.ndim == 2 and p.shape == (num_ntn, num_ntn_ant):
        return p

    raise ValueError(
        "user_powers must be None, shape (num_ntn,), or shape (num_ntn, num_ntn_ant)."
    )


def _covariance_from_static_channels(
    hi: np.ndarray,
    user_powers_2d: np.ndarray,
    noise_var: float,
) -> np.ndarray:
    """Build covariance analytically for static narrowband channels.

    Model:
        x = sum_{u,r} sqrt(p_{u,r}) h_{u,r} s_{u,r} + w
        E[s s^H] = I, E[w w^H] = noise_var * I
    """
    num_ntn, num_ntn_ant, num_bs_ant = hi.shape
    rxx = np.zeros((num_bs_ant, num_bs_ant), dtype=np.complex128)

    for u in range(num_ntn):
        for r in range(num_ntn_ant):
            h = hi[u, r, :].reshape(-1, 1)
            rxx += user_powers_2d[u, r] * (h @ h.conj().T)

    if noise_var > 0.0:
        rxx += noise_var * np.eye(num_bs_ant, dtype=np.complex128)
    return rxx


def _sample_covariance_from_snapshots(
    hi: np.ndarray,
    user_powers_2d: np.ndarray,
    noise_var: float,
    num_snapshots: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic snapshots and return (Rxx, X).

    - No QAM order is required here; symbols are circular Gaussian CN(0, 1).
    - This is standard for subspace estimation and is modulation-agnostic.
    """
    num_ntn, num_ntn_ant, num_bs_ant = hi.shape
    ur = num_ntn * num_ntn_ant

    # H shape: (M, UR), each column is one source channel vector to BS array.
    h_mat = hi.reshape(ur, num_bs_ant).T
    p_vec = user_powers_2d.reshape(ur)
    p_sqrt = np.sqrt(np.maximum(p_vec, 0.0))

    s = (
        rng.standard_normal((ur, num_snapshots))
        + 1j * rng.standard_normal((ur, num_snapshots))
    ) / np.sqrt(2.0)
    s *= p_sqrt[:, None]

    x_clean = h_mat @ s
    if noise_var > 0.0:
        w = (
            rng.standard_normal((num_bs_ant, num_snapshots))
            + 1j * rng.standard_normal((num_bs_ant, num_snapshots))
        ) * np.sqrt(noise_var / 2.0)
        x = x_clean + w
    else:
        x = x_clean

    rxx = (x @ x.conj().T) / float(num_snapshots)
    return rxx, x


def _estimate_num_sources_mdl(
    eigenvalues_desc: np.ndarray,
    num_snapshots: int,
    max_sources: Optional[int] = None,
) -> int:
    """Estimate source count with Wax-Kailath MDL."""
    eig = np.real(np.asarray(eigenvalues_desc, dtype=np.float64))
    m = eig.shape[0]
    if max_sources is None:
        max_sources = m - 1
    max_sources = int(np.clip(max_sources, 0, m - 1))

    # MDL is meaningful with at least a few snapshots.
    n = int(max(num_snapshots, m + 1))
    eps = 1e-12

    mdl_vals = np.full(max_sources + 1, np.inf, dtype=np.float64)
    for k in range(max_sources + 1):
        noise_eigs = np.maximum(eig[k:], eps)
        p = m - k
        if p <= 0:
            continue
        gm = np.exp(np.mean(np.log(noise_eigs)))
        am = np.mean(noise_eigs)
        if am <= eps:
            continue
        # Wax-Kailath MDL:
        # MDL(k) = -n*(m-k)*log(gm/am) + 0.5*k*(2m-k)*log(n)
        mdl_vals[k] = -n * p * np.log(gm / am) + 0.5 * k * (2 * m - k) * np.log(n)

    return int(np.argmin(mdl_vals))


def _estimate_num_sources_energy(
    eigenvalues_desc: np.ndarray,
    energy_ratio: float = 0.95,
) -> int:
    """Fallback source-count estimate by cumulative eigen-energy."""
    eig = np.real(np.asarray(eigenvalues_desc, dtype=np.float64))
    eig = np.maximum(eig, 0.0)
    total = float(np.sum(eig))
    if total <= 0.0:
        return 0
    csum = np.cumsum(eig) / total
    k = int(np.searchsorted(csum, energy_ratio) + 1)
    return int(np.clip(k, 0, eig.shape[0] - 1))


def _compute_user_scores(
    hi: np.ndarray,
    us: np.ndarray,
    en: np.ndarray,
    reduce_ntn_ant: Literal["max", "mean"],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute MUSIC-like user scores from signal/noise subspace projections.

    score = ||Us^H h||^2 / ||En^H h||^2 for normalized h.
    """
    num_ntn, num_ntn_ant, _ = hi.shape
    eps = 1e-12
    score_per_ant = np.zeros((num_ntn, num_ntn_ant), dtype=np.float64)

    for u in range(num_ntn):
        for r in range(num_ntn_ant):
            h = hi[u, r, :].reshape(-1, 1)
            norm_h = np.linalg.norm(h)
            if norm_h <= eps:
                score_per_ant[u, r] = 0.0
                continue
            h_n = h / norm_h

            sig_proj = np.linalg.norm(us.conj().T @ h_n) ** 2 if us.size else 0.0
            noi_proj = np.linalg.norm(en.conj().T @ h_n) ** 2 if en.size else eps
            score_per_ant[u, r] = float(sig_proj / max(noi_proj, eps))

    if reduce_ntn_ant == "max":
        score_user = np.max(score_per_ant, axis=1)
    elif reduce_ntn_ant == "mean":
        score_user = np.mean(score_per_ant, axis=1)
    else:
        raise ValueError("reduce_ntn_ant must be 'max' or 'mean'.")

    return score_per_ant, score_user


def detect_ntn_music_from_hi(
    hi: np.ndarray,
    *,
    num_sources: Optional[int] = None,
    threshold: Optional[float] = 1.0,
    user_powers: Optional[np.ndarray] = None,
    noise_var: float = 0.0,
    covariance_mode: Literal["analytic", "sample"] = "analytic",
    num_snapshots: int = 200,
    rng_seed: Optional[int] = None,
    source_estimation: Literal["mdl", "energy"] = "mdl",
    energy_ratio: float = 0.95,
    reduce_ntn_ant: Literal["max", "mean"] = "max",
) -> Dict[str, np.ndarray]:
    """Run narrowband MUSIC detection for one BS/sector channel tensor.

    Parameters
    ----------
    hi : np.ndarray
        Shape (num_ntn, num_ntn_ant, num_bs_ant), complex channel tensor.
    num_sources : int | None
        Signal subspace dimension K. If None, estimated automatically.
    threshold : float | None
        Detection threshold on user score.
        - If provided: detected_mask_user = score_user >= threshold
        - If None: top-K users are marked as detected.
    user_powers : np.ndarray | None
        Optional source powers:
        - shape (num_ntn,)
        - or shape (num_ntn, num_ntn_ant)
    noise_var : float
        Noise variance per BS antenna.
    covariance_mode : {"analytic", "sample"}
        "analytic": Rxx from static channels directly.
        "sample": synthetic snapshots are generated and sample covariance is used.
    num_snapshots : int
        Number of snapshots when covariance_mode == "sample".
    rng_seed : int | None
        Random seed for reproducibility in sample mode.
    source_estimation : {"mdl", "energy"}
        Automatic K-estimation rule when num_sources is None.
    energy_ratio : float
        Used only when source_estimation == "energy".
    reduce_ntn_ant : {"max", "mean"}
        How to aggregate multi-antenna NTN scores to user-level score.

    Returns
    -------
    Dict[str, np.ndarray]
        Keys:
        - "detected_mask_user": bool array, shape (num_ntn,)
        - "detected_mask_per_ant": bool array, shape (num_ntn, num_ntn_ant)
        - "score_user": float array, shape (num_ntn,)
        - "score_per_ant": float array, shape (num_ntn, num_ntn_ant)
        - "num_sources_est": int scalar in ndarray
        - "threshold_used": float scalar in ndarray
        - "eigenvalues_desc": float array, shape (num_bs_ant,)
        - "covariance": complex array, shape (num_bs_ant, num_bs_ant)
    """
    hi_c = _as_complex_array(hi)
    if hi_c.ndim != 3:
        raise ValueError(
            "hi must be a 3D tensor with shape (num_ntn, num_ntn_ant, num_bs_ant)."
        )

    num_ntn, num_ntn_ant, num_bs_ant = hi_c.shape
    if num_bs_ant < 2:
        raise ValueError("MUSIC requires num_bs_ant >= 2.")
    if noise_var < 0.0:
        raise ValueError("noise_var must be >= 0.")
    if covariance_mode == "sample" and num_snapshots < 2:
        raise ValueError("num_snapshots must be >= 2 in sample mode.")

    p_2d = _broadcast_powers(user_powers, num_ntn=num_ntn, num_ntn_ant=num_ntn_ant)

    if covariance_mode == "analytic":
        rxx = _covariance_from_static_channels(hi_c, p_2d, noise_var)
        n_for_mdl = max(num_ntn * num_ntn_ant, num_bs_ant + 1)
    elif covariance_mode == "sample":
        rng = np.random.default_rng(rng_seed)
        rxx, _x = _sample_covariance_from_snapshots(
            hi=hi_c,
            user_powers_2d=p_2d,
            noise_var=noise_var,
            num_snapshots=num_snapshots,
            rng=rng,
        )
        n_for_mdl = int(num_snapshots)
    else:
        raise ValueError("covariance_mode must be 'analytic' or 'sample'.")

    # Hermitian eigendecomposition.
    evals, evecs = np.linalg.eigh(rxx)
    idx = np.argsort(np.real(evals))[::-1]
    evals_desc = np.real(evals[idx])
    evecs_desc = evecs[:, idx]

    if num_sources is None:
        if source_estimation == "mdl":
            k_est = _estimate_num_sources_mdl(
                eigenvalues_desc=evals_desc, num_snapshots=n_for_mdl
            )
        elif source_estimation == "energy":
            k_est = _estimate_num_sources_energy(
                eigenvalues_desc=evals_desc, energy_ratio=energy_ratio
            )
        else:
            raise ValueError("source_estimation must be 'mdl' or 'energy'.")
    else:
        k_est = int(num_sources)

    k_est = int(np.clip(k_est, 0, num_bs_ant - 1))
    us = evecs_desc[:, :k_est] if k_est > 0 else np.empty((num_bs_ant, 0), dtype=np.complex128)
    en = evecs_desc[:, k_est:] if k_est < num_bs_ant else np.empty((num_bs_ant, 0), dtype=np.complex128)

    score_per_ant, score_user = _compute_user_scores(
        hi=hi_c,
        us=us,
        en=en,
        reduce_ntn_ant=reduce_ntn_ant,
    )

    if threshold is None:
        # If no threshold is provided, mark top-K users as detected.
        k_users = int(np.clip(k_est, 0, num_ntn))
        detected = np.zeros(num_ntn, dtype=bool)
        if k_users > 0:
            top_idx = np.argsort(score_user)[::-1][:k_users]
            detected[top_idx] = True
        detected_per_ant = np.repeat(detected[:, None], num_ntn_ant, axis=1)
        threshold_used = np.nan
    else:
        threshold_used = float(threshold)
        detected = score_user >= threshold_used
        detected_per_ant = score_per_ant >= threshold_used

    return {
        "detected_mask_user": detected,
        "detected_mask_per_ant": detected_per_ant,
        "score_user": score_user,
        "score_per_ant": score_per_ant,
        "num_sources_est": np.array(k_est, dtype=np.int64),
        "threshold_used": np.array(threshold_used, dtype=np.float64),
        "eigenvalues_desc": evals_desc,
        "covariance": rxx,
    }
