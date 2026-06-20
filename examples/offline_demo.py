"""Dependency-free demo of the metric (numpy only, no model, no torch).

    python examples/offline_demo.py
"""

import numpy as np

from question_decisiveness import decisiveness, decisiveness_raw, predict_matrix_caseV

# A strongly-ordered model (0 > 1 > 2 > 3) is decisive; near-equal utilities are not.
decisive_mu = np.array([3.0, 1.0, -1.0, -3.0])
indecisive_mu = np.array([0.1, 0.0, -0.05, 0.05])

print("decisive   mu ->", round(decisiveness(decisive_mu), 3))     # ~0.9
print("indecisive mu ->", round(decisiveness(indecisive_mu), 3))   # ~0.05

print("\npairwise P(i>j) for the decisive model:")
print(np.round(predict_matrix_caseV(decisive_mu), 2))

# Raw form straight from observed pair probabilities.
rows = [
    {"i": 0, "j": 1, "p_util": 0.95},
    {"i": 0, "j": 2, "p_util": 0.99},
    {"i": 1, "j": 2, "p_util": 0.90},
]
print("\ndecisiveness_raw ->", round(decisiveness_raw(rows), 3))
