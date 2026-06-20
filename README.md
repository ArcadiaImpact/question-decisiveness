# question-decisiveness

Measure how **decisive** (opinionated) a language model is over a set of options.

Given a list of items, the model is asked, for every pair, *"do you feel more
positively about A or B?"*. Its answer probability is read straight from the
next-token logprobs (no sampling). Those pairwise preferences are folded into a
1-D latent utility per item (Thurstone Case V), and **decisiveness** is how far
those preferences sit from a coin flip, on average:

- **0.0** — indifferent: every pair is ~50/50.
- **1.0** — certain: every pair is ~0/100.

The latent-utility model is Thurstone's Case V; see *Notes* for references.

## How it works

```
items (list[str])
  └─ elicit_logprobs()   prefill "<answer>", softmax over the A/B token logits
        → {(i, j): P(pick i | A=i, B=j)}            # one read per ordered pair
  └─ combine_orderings() average (i,j) and (j,i) to cancel slot/position bias
        → pref[i, j] = P(item i preferred over j)   # symmetric matrix
  └─ pref_to_edges() + fit_caseV_mle()  P(i>j) = Phi((mu_i - mu_j)/sqrt2)
        → mu   (latent utility per item)
  └─ decisiveness(mu)  mean |2P - 1| over all unordered pairs   → score in [0,1]
```

The metric itself (`decisiveness`, `predict_matrix_caseV`, `decisiveness_raw`) is
**pure numpy**. `torch` + `transformers` are only needed to fit utilities and to
elicit from a real model.

## Install

```bash
pip install -e .            # metric only (numpy)
pip install -e ".[model]"   # + torch/transformers for elicitation & fitting
```

## Use

End-to-end on a real model:

```python
from question_decisiveness import (
    load_model, elicit_logprobs, combine_orderings, pref_to_edges,
    fit_caseV_mle, decisiveness,
)

tok, model = load_model("Qwen/Qwen2.5-0.5B-Instruct")
items = ["pizza", "broccoli", "chocolate", "liver"]

ordered = elicit_logprobs(tok, model, items)     # {(i,j): P(pick i)}
pref    = combine_orderings(len(items), ordered)  # position-bias-free P(i>j)
mu      = fit_caseV_mle(pref_to_edges(pref), n=len(items))["mu"]

print(decisiveness(mu))   # e.g. 0.71
```

If you already have `mu` (or just a preference matrix), skip straight to the
metric — see `examples/offline_demo.py` (no model, no torch).

```bash
python examples/offline_demo.py     # numpy only
python examples/run_model.py        # downloads a small HF model
pytest                              # tests (fit test auto-skips without torch)
```

## The two metric forms

- `decisiveness(mu)` — headline metric over the **fitted** Case V matrix. Smooth,
  and defined for every pair (not just the ones you measured).
- `decisiveness_raw(rows)` — model-free diagnostic straight from the observed
  `p_util` values. Resolution-limited and noisier; handy as a sanity check.

## Notes / scope

- This library is the single-question forced-choice path, kept deliberately
  small. `fit_caseV_mle` also accepts sample-mode edges
  (`{"i","j","mode":"sample","wins_i","wins_j"}`) if you'd rather read
  preferences from sampled win counts than from logprobs.
- Natural extensions not included here: multiple question framings, richer
  consistency metrics (transitivity, unidimensionality R², bootstrap CIs), and
  4/8-bit / multi-GPU model loading (`load_model` here is the bf16 path).
- The latent-utility model is Thurstone's *Case V* (L. L. Thurstone, "A Law of
  Comparative Judgment", 1927): `P(i > j) = Φ((μ_i − μ_j)/√2)` with unit
  per-item variance.
