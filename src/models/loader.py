import logging
from typing import Tuple

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.utils import logging as hf_logging

logger = logging.getLogger(__name__)


def load_vllm_model_and_tokenizer(
    model_id: str,
) -> Tuple[object, PreTrainedTokenizerBase]:
    """Load the vLLM backend and the matching tokenizer."""

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    logger.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if not torch.cuda.is_available():
        raise RuntimeError("vLLM backend requires CUDA; no GPU detected.")

    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError(
            "vLLM backend requested but vllm is not installed. Install with `uv add vllm`."
        ) from exc

    if torch.cuda.is_bf16_supported():
        vllm_dtype = "bfloat16"
    else:
        vllm_dtype = "float16"

    logger.info("Loading vLLM model: %s (dtype=%s)", model_id, vllm_dtype)
    model = LLM(model=model_id, dtype=vllm_dtype)
    return model, tokenizer
