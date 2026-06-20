"""The "decisiveness" metric and the Thurstone Case V model it sits on.

Decisiveness = how *opinionated* a model is over a set of options. Given its
preferences between pairs of items, how far from indifference (a coin-flip) are
those preferences on average?

    0.0  → always indifferent (every pair ~50/50)
    1.0  → always certain     (every pair ~0/100)

The per-pair quantity is |2P - 1|, the distance of a preference probability P
from 0.5, rescaled to [0, 1].

Two forms:

decisiveness(mu)
    The headline metric. From fitted 1-D latent utilities `mu`, reconstruct the
    full pairwise preference matrix (Case V) and average |2P - 1| over all
    unordered pairs. Smooth, and defined for every pair — not just measured ones.

decisiveness_raw(rows)
    Model-free diagnostic straight from observed pairwise probabilities
    (`p_util`). Resolution-limited and noisier; useful as a sanity check.

Case V model:  P(i preferred over j) = Phi((mu_i - mu_j) / sqrt(2))
where Phi is the standard normal CDF and `mu` is the per-item latent utility.
The metric itself is pure-numpy; `fit_caseV_mle` (edges -> mu) uses torch.
"""

from __future__ import annotations

import math

import numpy as np

_SQRT2 = math.sqrt(2.0)

# Vectorized standard-normal CDF, Phi(x) = 0.5 * (1 + erf(x / sqrt(2))).
# (scipy.special.ndtr is faster/more accurate if you have scipy.)
_erf = np.vectorize(math.erf, otypes=[np.float64])


def _phi(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=np.float64) / _SQRT2))


def predict_matrix_caseV(mu) -> np.ndarray:
    """Full pairwise preference matrix from latent utilities `mu`.

    P[i, j] = Phi((mu_i - mu_j) / sqrt2), the Case V probability that item i is
    preferred over item j. The diagonal is set to 0.5.
    """
    mu = np.asarray(mu, dtype=np.float64)
    diff = mu[:, None] - mu[None, :]
    P = _phi(diff / _SQRT2)
    np.fill_diagonal(P, 0.5)
    return P


def decisiveness(mu) -> float:
    """Headline metric: mean |2*Phi - 1| over unordered pairs of the Case V
    matrix. Bounded [0, 1]."""
    P = predict_matrix_caseV(mu)
    iu = np.triu_indices(P.shape[0], k=1)        # unordered pairs (i < j)
    return float(np.mean(np.abs(2 * P[iu] - 1)))


def decisiveness_raw(rows) -> float:
    """Raw diagnostic: mean |2*p_util - 1| over observed edges (resolution-limited).

    `rows` is a list of dicts each carrying `p_util` = observed P(item i preferred
    over item j) for one measured pair.
    """
    p = np.array([float(r["p_util"]) for r in rows])
    return float(np.mean(np.abs(2 * p - 1))) if len(p) else float("nan")


def fit_caseV_mle(rows, n, steps=2000, lr=0.05, seed=0, device=None, mu_init=None) -> dict:
    """Homoscedastic Thurstone Case V MLE on `mu` (sigma fixed = 1), no prior.

        P(i > j) = Phi((mu_i - mu_j) / sqrt2)

    `rows` is a list of edge dicts:
        {"i": int, "j": int, "p_util": float}                              # prob mode
        {"i": int, "j": int, "mode": "sample", "wins_i": n, "wins_j": m}   # count mode
    `i`, `j` are item indices in [0, n); `p_util` = P(item i preferred over j).
    Gauge: `mu` is centered (mean 0). Returns {"mu": np.ndarray of shape (n,)}.
    Requires torch.
    """
    import torch

    def _phi_t(x):
        return 0.5 * (1.0 + torch.erf(x / _SQRT2))

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    i_idx, j_idx, w_pos, w_neg = [], [], [], []
    for r in rows:
        i_idx.append(int(r["i"]))
        j_idx.append(int(r["j"]))
        if r.get("mode") == "sample":
            w_pos.append(float(r["wins_i"]))
            w_neg.append(float(r["wins_j"]))
        else:
            p = float(r["p_util"])
            w_pos.append(p)
            w_neg.append(1.0 - p)

    ii = torch.as_tensor(np.asarray(i_idx), device=device)
    jj = torch.as_tensor(np.asarray(j_idx), device=device)
    wp = torch.as_tensor(np.asarray(w_pos, dtype=np.float64), device=device)
    wn = torch.as_tensor(np.asarray(w_neg, dtype=np.float64), device=device)

    if mu_init is None:
        mu = torch.zeros(n, dtype=torch.float64, device=device)
    else:
        mu = torch.as_tensor(np.asarray(mu_init), dtype=torch.float64, device=device).clone()
    mu.requires_grad_(True)

    opt = torch.optim.Adam([mu], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        P = _phi_t((mu[ii] - mu[jj]) / _SQRT2).clamp(1e-9, 1 - 1e-9)
        nll = -(wp * torch.log(P) + wn * torch.log1p(-P)).sum()
        nll.backward()
        opt.step()
    with torch.no_grad():
        mu_c = mu - mu.mean()                       # additive gauge
    return {"mu": mu_c.detach().cpu().numpy()}
