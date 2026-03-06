import logging
import time
from argparse import Namespace
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)

FIM_PREFIX = "<fim_prefix>"
FIM_SUFFIX = "<fim_suffix>"
FIM_MIDDLE = "<fim_middle>"
FIM_PAD = "<fim_pad>"
END_OF_TEXT = "<|endoftext|>"
FILE_SEP = "<file_sep>"


def create_infill_prompt(code: str, element: Dict, filepath: str = "") -> str:
    """
    Create FIM prompt for infilling by masking the element.

    Uses character indices only: delimiters and punctuation are naturally preserved
    in prefix/suffix by slicing the original code.

    Constructs StarCoder2 canonical FIM format:
    <fim_prefix>[code_before_gap]<fim_suffix>[code_after_gap]<fim_middle>

    Args:
        code: Full code string
        element: Element dict with start, end, value

    Returns:
        FIM prompt string
    """
    max_context = 3000
    prefix_start = max(0, element["start"] - max_context)
    suffix_end = element["end"] + max_context

    prefix = code[prefix_start:element["start"]]
    suffix = code[element["end"]:suffix_end]
    if prefix_start == 0:
        # We are at the true start of the file.
        # Inject <file_sep> and the filepath (if available) for maximum memorization recall!
        path_str = f"{filepath}\n" if filepath and filepath.strip() else ""
        prompt = f"{FILE_SEP}{FIM_PREFIX}{path_str}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
    else:
        # We are starting in the middle of a file chunk. Do NOT use <file_sep>.
        prompt = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
    return prompt


def run_infill(
    model,
    tokenizer,
    prompt: str,
    device: str,
    params: Namespace
) -> Optional[str]:
    """
    Run model infilling using FIM prompt.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Complete FIM prompt string
        device: Device to run on ('cuda' or 'cpu')
        params: Namespace with generation parameters (e.g. max_generated_tokens)

    Returns:
        Generated infill text
    """
    try:
        # Tokenize
        start = time.time()
        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        ).to(device)

        # Check if input is too long
        if inputs.input_ids.shape[1] + params.max_generated_tokens > 2048:
            logger.warning(f"    Input too long: {inputs.input_ids.shape[1]} tokens + {params.max_generated_tokens} max_tokens > 2048")
            return "too_many_tokens"

        gen_start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=params.max_generated_tokens,
                do_sample=True,
                temperature=0.2,
                top_p=0.95,
                pad_token_id=tokenizer.convert_tokens_to_ids(FIM_PAD),
                eos_token_id=[tokenizer.convert_tokens_to_ids(END_OF_TEXT), tokenizer.convert_tokens_to_ids(FILE_SEP)],
            )
        gen_time = time.time() - gen_start
        logger.info(f"    Model generation took {gen_time:.3f}s, output shape: {outputs.shape}")

        # Decode
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract generated infill after <fim_middle>
        if FIM_MIDDLE in full_output:
            start_idx = full_output.find(FIM_MIDDLE) + len(FIM_MIDDLE)
            end_idx = full_output.find(END_OF_TEXT, start_idx)
            if end_idx == -1:
                end_idx = len(full_output)
            return full_output[start_idx:end_idx].strip()

        return None

    except Exception as e:
        logger.warning(f"    Error during infill: {e}")
        return None