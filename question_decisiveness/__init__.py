"""question-decisiveness: measure how opinionated an LM is over a set of options.

End-to-end:
    from question_decisiveness import (
        load_model, elicit_logprobs, combine_orderings, pref_to_edges,
        fit_caseV_mle, decisiveness,
    )
    tok, model = load_model("Qwen/Qwen2.5-0.5B-Instruct")
    items = ["pizza", "broccoli", "ice cream", "liver"]
    ordered = elicit_logprobs(tok, model, items)        # {(i,j): P(pick i)}
    pref    = combine_orderings(len(items), ordered)     # position-bias-free P(i>j)
    edges   = pref_to_edges(pref)
    mu      = fit_caseV_mle(edges, n=len(items))["mu"]
    score   = decisiveness(mu)                            # in [0, 1]
"""

from .prompts import ASSISTANT_PREFIX, build_prompt, parse_answer
from .elicit import (
    load_model,
    compare_pairs,
    elicit_logprobs,
    combine_orderings,
    pref_to_edges,
)
from .decisiveness import (
    decisiveness,
    decisiveness_raw,
    predict_matrix_caseV,
    fit_caseV_mle,
)

__all__ = [
    "ASSISTANT_PREFIX", "build_prompt", "parse_answer",
    "load_model", "compare_pairs", "elicit_logprobs",
    "combine_orderings", "pref_to_edges",
    "decisiveness", "decisiveness_raw", "predict_matrix_caseV", "fit_caseV_mle",
]
