import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from transformers.utils import logging as hf_logging

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_id: str) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    # Suppress transformers' logging and progress bars
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    logger.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("Before:")
    print(tokenizer.special_tokens_map)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print("After:")
    print(tokenizer.special_tokens_map)

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
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch_dtype,
    )

    model.eval()
    return model, tokenizer
