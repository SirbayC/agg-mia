import logging
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def is_vllm_model(model) -> bool:
    return model.__class__.__module__.startswith("vllm")


def truncate_text(tokenizer, text: str, max_length: int) -> Tuple[str, List[int]]:
    tokenized = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    token_ids = tokenized["input_ids"][0].tolist()
    return tokenizer.decode(token_ids, skip_special_tokens=False), token_ids


def _entry_logprob(entry) -> Optional[float]:
    if entry is None:
        return None
    if hasattr(entry, "logprob"):
        return float(entry.logprob)
    try:
        return float(entry)
    except (TypeError, ValueError):
        return None


def _selected_token_logprob(position_logprobs, token_id: int) -> Optional[float]:
    if position_logprobs is None:
        return None

    if hasattr(position_logprobs, "get"):
        entry = position_logprobs.get(token_id)
        logprob = _entry_logprob(entry)
        if logprob is not None:
            return logprob

    try:
        entry = position_logprobs[token_id]
        logprob = _entry_logprob(entry)
        if logprob is not None:
            return logprob
    except Exception:
        pass

    candidate_ids = getattr(position_logprobs, "token_ids", None)
    candidate_logprobs = getattr(position_logprobs, "logprobs", None)
    if candidate_ids is not None and candidate_logprobs is not None:
        for index, candidate_id in enumerate(candidate_ids):
            if candidate_id == token_id and index < len(candidate_logprobs):
                try:
                    return float(candidate_logprobs[index])
                except (TypeError, ValueError):
                    return None

    return None


def _request_prompt_logprobs(model, prompt: str, token_ids: Sequence[int]):
    from vllm import SamplingParams

    base_kwargs = dict(
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        prompt_logprobs=1,
    )

    try:
        sampling_params = SamplingParams(**base_kwargs, logprob_token_ids=list(token_ids))
    except TypeError:
        sampling_params = SamplingParams(**base_kwargs)

    outputs = model.generate(prompt, sampling_params=sampling_params)
    if not outputs:
        return [], []

    request_output = outputs[0]
    prompt_logprobs = getattr(request_output, "prompt_logprobs", None) or []
    prompt_token_ids = getattr(request_output, "prompt_token_ids", None) or list(token_ids)
    return prompt_logprobs, prompt_token_ids


def get_prompt_token_logprobs(model, tokenizer, text: str, max_length: int) -> np.ndarray:
    prompt, token_ids = truncate_text(tokenizer, text, max_length)
    if len(token_ids) < 2:
        return np.array([])

    prompt_logprobs, prompt_token_ids = _request_prompt_logprobs(model, prompt, token_ids)

    selected_logprobs = []
    for position, token_id in enumerate(prompt_token_ids[1:], start=1):
        if position >= len(prompt_logprobs):
            break
        logprob = _selected_token_logprob(prompt_logprobs[position], token_id)
        if logprob is not None:
            selected_logprobs.append(logprob)

    return np.array(selected_logprobs)


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_input_tokens: Optional[int],
    max_generated_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool = True,
    top_k: Optional[int] = None,
    stop_token_ids: Optional[Iterable[int]] = None,
) -> str:
    from vllm import SamplingParams

    if max_input_tokens is not None:
        prompt_text, _ = truncate_text(tokenizer, prompt, max_input_tokens)
    else:
        prompt_text = prompt

    filtered_stop_token_ids = [token_id for token_id in (stop_token_ids or []) if token_id is not None]
    top_k_value = top_k if do_sample and top_k is not None else -1
    sampling_params = SamplingParams(
        max_tokens=max_generated_tokens,
        temperature=temperature if do_sample else 0.0,
        top_p=top_p if do_sample else 1.0,
        top_k=top_k_value,
        stop_token_ids=filtered_stop_token_ids or None,
    )
    outputs = model.generate(prompt_text, sampling_params=sampling_params)
    if not outputs or not outputs[0].outputs:
        return ""
    return outputs[0].outputs[0].text