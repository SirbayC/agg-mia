import logging
import os
from importlib.util import find_spec
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers.utils import logging as hf_logging

logger = logging.getLogger(__name__)


def _resolve_attention_backend(requested: str) -> Optional[str]:
    """Resolve attention backend from runtime capabilities.

    requested values:
    - auto: flash_attention_2 when available, otherwise sdpa on CUDA.
    - flash_attention_2: force FA2 (falls back to sdpa if unavailable).
    - sdpa | eager: force backend.
    """
    requested = requested.strip().lower()
    valid = {"auto", "flash_attention_2", "sdpa", "eager"}
    if requested not in valid:
        logger.warning("Unknown attn_implementation=%s, using auto", requested)
        requested = "auto"

    if not torch.cuda.is_available():
        return None

    if requested in {"sdpa", "eager"}:
        return requested

    # flash-attn2 supports Ampere+ (e.g., A100) and requires installed package.
    has_flash_attn = find_spec("flash_attn") is not None
    device_capability = torch.cuda.get_device_capability(0)
    ampere_or_newer = device_capability[0] >= 8

    if has_flash_attn and ampere_or_newer:
        return "flash_attention_2"

    if requested == "flash_attention_2":
        logger.warning(
            "AGG_ATTN_IMPL=flash_attention_2 requested, but unavailable "
            "(installed=%s, capability=%s). Falling back to sdpa.",
            has_flash_attn,
            device_capability,
        )

    return "sdpa"


def load_model_and_tokenizer(
    model_id: str,
    attn_implementation: str = "auto",
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    # Suppress transformers' logging and progress bars
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    logger.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model: %s", model_id)
    
    # Determine best dtype: bf16 > fp16 > fp32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
        logger.info("Using bfloat16 precision")
    elif torch.cuda.is_available():
        torch_dtype = torch.float16
        logger.info("Using float16 precision")
    else:
        torch_dtype = torch.float32
        logger.info("Using float32 precision")

    if torch.cuda.is_available() and os.getenv("AGG_ALLOW_TF32", "1") == "1":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled for CUDA matmul/cuDNN")

    attn_backend = _resolve_attention_backend(attn_implementation)
    if attn_backend is not None:
        logger.info("Attention backend: %s", attn_backend)
    else:
        logger.info("Attention backend: default (CPU)")

    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if attn_backend is not None:
        model_kwargs["attn_implementation"] = attn_backend
    
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    model.eval()
    return model, tokenizer
