import logging
import os
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers.utils import logging as hf_logging

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_id: str,
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

    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    model.eval()
    return model, tokenizer


def load_vllm_model_and_tokenizer(
    model_id: str,
) -> Tuple[object, PreTrainedTokenizerBase]:
    """Load vLLM model plus HF tokenizer for prompt/token utilities.

    vLLM handles generation while tokenizer is still used by TraWiC for
    input-length checks and stop-token IDs.
    """
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
