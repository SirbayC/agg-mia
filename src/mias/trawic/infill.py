import logging
import time
from typing import Dict, Optional

import torch # type: ignore

from .config import TraWiCParams

logger = logging.getLogger(__name__)

FIM_PREFIX = "<fim_prefix>"
FIM_SUFFIX = "<fim_suffix>"
FIM_MIDDLE = "<fim_middle>"
FIM_PAD = "<fim_pad>"
END_OF_TEXT = "<|endoftext|>"
FILE_SEP = "<file_sep>"


def create_infill_prompt(code: str, element: Dict, params: TraWiCParams, filepath: str = "") -> str:
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
    prefix_start = max(0, element["start"] - params.max_context)
    suffix_end = element["end"] + params.max_context

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
    params: TraWiCParams,
    element_type: str = "",
    element: Optional[Dict] = None
) -> Optional[str]:
    """
    Run model infilling using FIM prompt.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Complete FIM prompt string
        device: Device to run on ('cuda' or 'cpu')
        params: TraWiCParams with generation parameters
        element_type: Type of element being infilled (e.g., "function_names", "strings")
        element: Element dict with metadata like quote_char

    Returns:
        Generated infill text
    """
    try:
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        ).to(device)

        # Check if input is too long
        if inputs.input_ids.shape[1] + params.max_generated_tokens > params.max_total_tokens:
            logger.warning(f"    Input too long: {inputs.input_ids.shape[1]} tokens + {params.max_generated_tokens} max_tokens > {params.max_total_tokens}")
            return "too_many_tokens"

        # Build custom stopping tokens based on element type
        stop_strings = []
        if element_type == "class_names":
            stop_strings = [":", "("]
        elif element_type == "function_names":
            stop_strings = ["("]
        elif element_type == "variable_names":
            stop_strings = ["="]
        elif element_type == "strings" and element and "quote_char" in element:
            stop_strings = [element["quote_char"]]
        elif element_type == "comments":
            stop_strings = ["\n"]
        elif element_type == "docstrings" and element and "quote_char" in element:
            stop_strings = [element["quote_char"]]
        
        # Convert stopping strings to token IDs
        eos_token_ids = [
            tokenizer.convert_tokens_to_ids(END_OF_TEXT),
            tokenizer.convert_tokens_to_ids(FILE_SEP)
        ]
        
        for stop_str in stop_strings:
            # Encode the stop string and get its token ID(s)
            stop_tokens = tokenizer.encode(stop_str, add_special_tokens=False)
            if stop_tokens:
                # Add the first token ID (for single-character stops, this should be sufficient)
                eos_token_ids.extend(stop_tokens)
        
        logger.debug(f"    Using stopping tokens for {element_type}: {stop_strings} -> IDs: {eos_token_ids}")

        gen_start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=params.max_generated_tokens,
                do_sample=True,
                temperature=params.temperature,
                top_p=params.top_p,
                pad_token_id=tokenizer.convert_tokens_to_ids(FIM_PAD),
                eos_token_id=eos_token_ids,
            )
        gen_time = time.time() - gen_start
        logger.debug(f"    Model generation took {gen_time:.3f}s, output shape: {outputs.shape}")

        # Decode
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Extract generated infill after <fim_middle>
        if FIM_MIDDLE in full_output:
            start_idx = full_output.find(FIM_MIDDLE) + len(FIM_MIDDLE)
            end_idx = len(full_output)
            stop_strings.append(END_OF_TEXT)  # Ensure we always stop at end of text if no other stop string is found
            stop_strings.append(FILE_SEP)  # Also stop at file separator to prevent bleeding into next file content
            for stop_str in stop_strings:
                stop_idx = full_output.find(stop_str, start_idx)
                if stop_idx != -1 and stop_idx < end_idx:
                    end_idx = stop_idx
            
            return full_output[start_idx:end_idx].strip()

        return None

    except Exception as e:
        logger.warning(f"    Error during infill: {e}")
        return None