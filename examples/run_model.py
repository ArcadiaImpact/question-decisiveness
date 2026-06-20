"""End-to-end: prompt a small HF model and score its decisiveness over a list.

    python examples/run_model.py            # default tiny model
    python examples/run_model.py Qwen/Qwen2.5-0.5B-Instruct

Needs `torch` + `transformers` (and a model download). For a dependency-free
demo of just the metric, see examples/offline_demo.py.
"""

import sys

from question_decisiveness import (
    load_model, elicit_logprobs, combine_orderings, pref_to_edges,
    fit_caseV_mle, decisiveness, decisiveness_raw,
)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
ITEMS = ["pizza", "broccoli", "chocolate", "liver", "strawberries"]

tok, model = load_model(MODEL)
ordered = elicit_logprobs(tok, model, ITEMS)        # {(i,j): P(pick i | A=i,B=j)}
pref = combine_orderings(len(ITEMS), ordered)        # P(i preferred over j)
edges = pref_to_edges(pref)
mu = fit_caseV_mle(edges, n=len(ITEMS))["mu"]

order = sorted(range(len(ITEMS)), key=lambda k: -mu[k])
print(f"model: {MODEL}")
print("ranking (most -> least preferred):")
for k in order:
    print(f"  {ITEMS[k]:<14} mu={mu[k]:+.3f}")
print(f"\ndecisiveness      = {decisiveness(mu):.3f}")
print(f"decisiveness_raw  = {decisiveness_raw(edges):.3f}")
