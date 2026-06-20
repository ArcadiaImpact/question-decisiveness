"""Score decisiveness for a model hosted on OpenRouter.

Same pipeline as run_model.py, but the A/B preference is read from the API's
token `logprobs` instead of local logits — so no GPU or model download, just an
API key. The downstream fit (`fit_caseV_mle`) still needs torch.

    export OPENROUTER_API_KEY=sk-or-...
    uv run examples/run_openrouter.py                         # default model
    uv run examples/run_openrouter.py meta-llama/llama-3.1-8b-instruct

Requires the `openrouter` extra:  uv pip install -e ".[openrouter]"

NOTE: the model must support `logprobs` / `top_logprobs` through OpenRouter
(most OpenAI and Llama models do; many others return no logprobs).
"""

import math
import os
import sys

from openai import OpenAI

from question_decisiveness import (
    combine_orderings, pref_to_edges, fit_caseV_mle, decisiveness, decisiveness_raw,
)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "openai/gpt-4o-mini"
ITEMS = ["pizza", "broccoli", "chocolate", "liver", "strawberries"]

PROMPT = ("Do you feel more positively about A: {a} or B: {b}? "
          "Reply with exactly one letter, A or B, and nothing else.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],   # raises if unset
)


def p_pick_a(top_logprobs) -> float:
    """P(pick A) from the first token's top_logprobs (softmax over the A/B mass)."""
    lp_a = lp_b = None
    for cand in top_logprobs:
        t = cand.token.strip().upper()
        if t == "A" and lp_a is None:
            lp_a = cand.logprob
        elif t == "B" and lp_b is None:
            lp_b = cand.logprob
    if lp_a is None and lp_b is None:
        return 0.5                                # model gave no A/B mass -> indifferent
    ea = math.exp(lp_a) if lp_a is not None else 0.0
    eb = math.exp(lp_b) if lp_b is not None else 0.0
    return ea / (ea + eb)


def compare_pair(i: int, j: int) -> float:
    """P(pick item i | A=i, B=j) for one ordered pair via the OpenRouter API."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT.format(a=ITEMS[i], b=ITEMS[j])}],
        max_tokens=1,
        temperature=0,
        logprobs=True,
        top_logprobs=20,
    )
    return p_pick_a(resp.choices[0].logprobs.content[0].top_logprobs)


n = len(ITEMS)
ordered = {(i, j): compare_pair(i, j)
           for i in range(n) for j in range(n) if i != j}     # {(i,j): P(pick i)}

pref = combine_orderings(n, ordered)                           # position-bias-free P(i>j)
edges = pref_to_edges(pref)
mu = fit_caseV_mle(edges, n=n)["mu"]

order = sorted(range(n), key=lambda k: -mu[k])
print(f"model: {MODEL} (via OpenRouter)")
print("ranking (most -> least preferred):")
for k in order:
    print(f"  {ITEMS[k]:<14} mu={mu[k]:+.3f}")
print(f"\ndecisiveness      = {decisiveness(mu):.3f}")
print(f"decisiveness_raw  = {decisiveness_raw(edges):.3f}")
