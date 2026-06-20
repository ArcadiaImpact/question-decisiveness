"""Prompt a HuggingFace causal LM and read its A/B preference from logprobs.

Pipeline:
    items (list of strings)
      -> for each ordered pair (i, j): render the forced-choice prompt, prefill
         ``<answer>``, take next-token logits, softmax over the A/B token ids
         -> P(pick slot A)
      -> combine_orderings(): average the (i,j) and (j,i) reads to cancel slot bias
         -> a symmetric preference matrix  pref[i, j] = P(item i preferred over j)
      -> pref_to_edges(): one edge per unordered pair, ready for the Case V fit.

The logit read (`compare_pairs`) is the heart of it: because the prompt is
prefilled with ``<answer>``, the next token is forced to A or B, so the softmax
over just those two token ids is a faithful read of the model's preference
probability — no sampling, no parsing.

`transformers`/`torch` are imported lazily so the metric code in
``decisiveness.py`` stays usable without a GPU or a model.
"""

from __future__ import annotations

import numpy as np

from .prompts import ASSISTANT_PREFIX, build_prompt


def load_model(model_id: str, dtype: str = "bfloat16", revision: str | None = None):
    """Load a causal LM + tokenizer for left-padded batched forced-choice reads.

    This is the bf16 path. ``device_map="auto"`` (which needs `accelerate`) shards
    larger models across GPUs and keeps everything on cuda:0 on a single GPU; when
    `accelerate` isn't installed it falls back to loading on one device.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_kwargs = {"revision": revision} if revision else {}
    tok = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype_t = getattr(torch, dtype)
    kwargs = {"revision": revision} if revision else {}
    try:
        import accelerate  # noqa: F401
        kwargs["device_map"] = "auto"
    except ImportError:
        pass

    # transformers renamed `torch_dtype` -> `dtype`; support both across versions.
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype_t, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype_t, **kwargs)

    if "device_map" not in kwargs:
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return tok, model


def _apply_chat(tok, messages, add_generation_prompt):
    """apply_chat_template with fallbacks for templates that reject
    enable_thinking, and for tokenizers with no chat template at all."""
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except Exception:
        try:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            parts = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
            if add_generation_prompt:
                parts.append("Assistant:")
            return "\n".join(parts)


def _prefill_text(tok, a: str, b: str, system_prompt: str | None = None) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": build_prompt(a, b)})
    text = _apply_chat(tok, messages, add_generation_prompt=True)
    return text + ASSISTANT_PREFIX


def _next_token_after_prefix(tok, token_text: str) -> int:
    """The single token id the model would emit for `token_text` right after the
    assistant prefix (handles tokenizers that merge the prefix with the letter)."""
    base = tok.encode(ASSISTANT_PREFIX, add_special_tokens=False)
    candidate = tok.encode(ASSISTANT_PREFIX + token_text, add_special_tokens=False)

    if candidate[: len(base)] == base:
        suffix = candidate[len(base):]
        if suffix:
            return suffix[0]
    for idx, token_id in enumerate(candidate):
        if idx >= len(base) or token_id != base[idx]:
            return token_id
    simple = tok.encode(token_text, add_special_tokens=False)
    if len(simple) == 1:
        return simple[0]
    raise ValueError(f"could not determine single token id for {token_text!r}")


def _ab_token_ids(tok) -> tuple[int, int]:
    return _next_token_after_prefix(tok, "A"), _next_token_after_prefix(tok, "B")


def _logits_from_output(output):
    if hasattr(output, "logits"):
        return output.logits
    logits_like = [value for key, value in vars(output).items() if "logits" in key]
    if logits_like:
        return logits_like[0]
    raise AttributeError("model output does not expose logits or logits-like fields")


def _model_input_device(model):
    import torch

    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compare_pairs(tok, model, items: list[str], pairs, batch_size: int = 64,
                  system_prompt: str | None = None) -> dict:
    """Return {(i, j): P(pick item i | A=i, B=j)} for explicit ordered pairs.

    P is read straight from the softmax over the A/B token logits at the position
    right after the ``<answer>`` prefill — no sampling. If `system_prompt` is
    given it is prepended to every prompt (e.g. to put a base model in a persona).
    """
    import torch

    pairs = list(pairs)
    a_id, b_id = _ab_token_ids(tok)
    ordered: dict[tuple[int, int], float] = {}
    device = _model_input_device(model)

    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            texts = [_prefill_text(tok, items[i], items[j], system_prompt=system_prompt)
                     for i, j in batch]
            enc = tok(texts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(device)
            logits = _logits_from_output(model(**enc))[:, -1, :]
            ab = torch.stack([logits[:, a_id], logits[:, b_id]], dim=-1)
            p_a = torch.softmax(ab.float(), dim=-1)[:, 0].cpu().numpy()
            for (i, j), pa in zip(batch, p_a):
                ordered[(i, j)] = float(pa)
    return ordered


def elicit_logprobs(tok, model, items: list[str], batch_size: int = 64,
                    system_prompt: str | None = None) -> dict:
    """Return {(i, j): P(pick item i | A=i, B=j)} over all ordered i != j pairs."""
    n = len(items)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    return compare_pairs(tok, model, items, pairs, batch_size=batch_size,
                         system_prompt=system_prompt)


def combine_orderings(n: int, ordered: dict[tuple[int, int], float]) -> np.ndarray:
    """Average the two slot orderings of each pair into a position-bias-free
    preference matrix.  pref[i, j] = P(item i preferred over item j), in [0, 1].

        pref[i, j] = 0.5 * (P(pick i | A=i,B=j) + (1 - P(pick j | A=j,B=i)))
    """
    pref = np.full((n, n), 0.5, dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p_ij = ordered[(i, j)]   # P(pick i | A=i, B=j)
            p_ji = ordered[(j, i)]   # P(pick j | A=j, B=i)
            pref[i, j] = 0.5 * (p_ij + (1.0 - p_ji))
    return pref


def pref_to_edges(pref: np.ndarray) -> list[dict]:
    """One edge per unordered pair (i < j): {"i", "j", "p_util"} for the Case V fit."""
    n = pref.shape[0]
    return [{"i": i, "j": j, "p_util": float(pref[i, j])}
            for i in range(n) for j in range(i + 1, n)]
