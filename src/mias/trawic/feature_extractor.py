"""
TraWiC Feature Extraction Pipeline

Extracts membership inference features by testing if a model can reconstruct
masked code elements (function names, variables, comments, etc.).
"""

import logging
import random
import re
import time
from typing import Dict, Optional

from fuzzywuzzy import fuzz
from .config import TraWiCParams
from .infill import create_infill_prompt, run_infill_batch

logger = logging.getLogger(__name__)

FIM_PREFIX = "<fim_prefix>"
FIM_SUFFIX = "<fim_suffix>"
FIM_MIDDLE = "<fim_middle>"
FIM_PAD = "<fim_pad>"
END_OF_TEXT = "<|endoftext|>"
FILE_SEP = "<file_sep>"
REPO = "<repo_name>"

def _extract_elements(code: str) -> Dict:
    """
    Extract code elements (function names, class names, variables, etc.) using regex.
    
    Args:
        code: Python code string
        
    Returns:
        Dict with extracted elements by type
    """
    elements = {
        "docstrings": [],
        "comments": [],
        "function_names": [],
        "class_names": [],
        "variable_names": [],
        "strings": []
    }
    
    try:
        # Extract docstrings (content only, supports both triple-double and triple-single quotes)
        for match in re.finditer(r'("""|\'\'\')([\s\S]*?)\1', code):
            elements["docstrings"].append({
                "value": match.group(2),  # Only the content, not the quotes
                "start": match.start(2),  # Position of content start
                "end": match.end(2),      # Position of content end
                "quote_char": match.group(1)  # """ or '''
            })
        
        # Extract comments (content only, after '#')
        for match in re.finditer(r"\s*#(.*)", code):
            comment_start = match.start(1)
            comment_end = match.end(1)
            elements["comments"].append({
                "value": code[comment_start:comment_end],
                "start": comment_start,
                "end": comment_end
            })
        
        # Extract function names
        for match in re.finditer(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([\s\S]*?)\)", code):
            elements["function_names"].append({
                "value": match.group(1),
                "start": match.start(1),
                "end": match.end(1)
            })
        
        # Extract class names
        for match in re.finditer(r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\(([\s\S]*?)\))?", code):
            elements["class_names"].append({
                "value": match.group(1),
                "start": match.start(1),
                "end": match.end(1)
            })
        
        # Extract variable names (assignments, with optional type annotations)
        for match in re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)(?:\s*:\s*[^=]*?)?\s*=", code):
            elements["variable_names"].append({
                "value": match.group(1),
                "start": match.start(1),
                "end": match.end(1)
            })
        
        # Extract strings (only the content inside single/double quotes)
        for match in re.finditer(r'(["\'])(.+?)\1', code):
            elements["strings"].append({
                "value": match.group(2),  # Only the content, not the quotes
                "start": match.start(2),  # Position of content start
                "end": match.end(2),      # Position of content end
                "quote_char": match.group(1)  # " or '
            })
    except Exception as e:
        logger.warning(f"Error extracting elements: {e}")
    
    return elements


def _check_similarity(target: str, output: Optional[str], threshold: int) -> int:
    """
    Check similarity between target and model output.
    
    Args:
        target: Ground truth element
        output: Model generated output
        metric: "exact" or "fuzzy"
        threshold: Similarity threshold
        
    Returns:
        1 if similarity >= threshold, 0 otherwise
    """
    if output is None or output == "too_many_tokens":
        return 0
    
    if threshold == 100: 
        return 1 if target.strip().upper() == output.strip().upper() else 0
    else:  # fuzzy
        similarity = fuzz.ratio(target.strip(), output.strip())
        return 1 if similarity >= threshold else 0


def extract_features(
    code: str,
    model,
    tokenizer,
    device: str,
    params: TraWiCParams
) -> Dict[str, float]:
    """
    Extract TraWiC features from a code sample.
    
    This is the main exported function that orchestrates the full pipeline:
    1. Extract code elements (functions, classes, variables, etc.)
    2. For each element, create an infill prompt by masking it
    3. Run the model to predict the masked element
    4. Check if prediction matches original (exact or fuzzy)
    5. Aggregate hit ratios as features
    
    Args:
        code: Python code string
        model: The language model
        tokenizer: The tokenizer
        device: Device to run on ('cuda' or 'cpu')
        params: Additional parameters (e.g., thresholds)
        
    Returns:
        Feature dict with normalized hit ratios for each element type:
        - class_hits: Proportion of class names the model reconstructed
        - class_nums_total: Total number of class names found
        - function_hits: Proportion of function names reconstructed
        - function_nums_total: Total function names
        - variable_hits: Proportion of variable names reconstructed
        - variable_nums_total: Total variables
        - string_hits: Proportion of strings reconstructed
        - string_nums_total: Total strings
        - comment_hits: Proportion of comments reconstructed
        - comment_nums_total: Total comments
        - docstring_hits: Proportion of docstrings reconstructed
        - docstring_nums_total: Total docstrings
    """
    features = {
        "class_hits": 0.0,
        "class_nums_total": 0,
        "function_hits": 0.0,
        "function_nums_total": 0,
        "variable_hits": 0.0,
        "variable_nums_total": 0,
        "string_hits": 0.0,
        "string_nums_total": 0,
        "comment_hits": 0.0,
        "comment_nums_total": 0,
        "docstring_hits": 0.0,
        "docstring_nums_total": 0,
    }
    
    # Extract all elements
    logger.debug(f"Extracting elements from code (length: {len(code)} chars)")
    start_time = time.time()
    elements = _extract_elements(code)
    extract_time = time.time() - start_time
    logger.debug(f"Element extraction took {extract_time:.3f}s")
    
    # Log element counts
    total_elements = sum(len(el) for el in elements.values())
    logger.debug(f"Extracted {total_elements} elements: {', '.join(f'{k}={len(v)}' for k, v in elements.items())}")
    
    # Collect all prompts to run as a batch (one model.generate() call per element type)
    batch_prompts = []
    batch_types = []
    batch_targets = []

    for level, element_list in elements.items():
        if len(element_list) > params.max_elements_per_type:
            element_list = random.sample(element_list, params.max_elements_per_type)
            logger.debug(f"Sampled {params.max_elements_per_type}/{len(elements[level])} {level} elements")

        for element in element_list:
            target = element["value"]
            if not target or len(target.strip()) < 2:
                logger.debug(f"  Skipping short {level} element: '{target[:20]}'")
                continue
            batch_prompts.append(create_infill_prompt(code, element, params))
            batch_types.append(level)
            batch_targets.append(target)

    # Run batch inference — one model.generate() per element type
    logger.debug(f"Running batch inference on {len(batch_prompts)} elements...")
    infer_start = time.time()
    outputs = run_infill_batch(batch_prompts, batch_types, model, tokenizer, device, params)
    logger.debug(f"Batch inference took {time.time() - infer_start:.2f}s")

    # Process results
    for level, target, output in zip(batch_types, batch_targets, outputs):
        threshold = (
            params.syntactic_threshold
            if level in ["function_names", "class_names", "variable_names"]
            else params.semantic_threshold
        )
        hit = _check_similarity(target, output, threshold)
        logger.debug(f"  {level}: '{target[:30]}' -> '{str(output)[:30]}' | hit={hit}")

        if level == "class_names":
            features["class_hits"] += hit
            features["class_nums_total"] += 1
        elif level == "function_names":
            features["function_hits"] += hit
            features["function_nums_total"] += 1
        elif level == "variable_names":
            features["variable_hits"] += hit
            features["variable_nums_total"] += 1
        elif level == "strings":
            features["string_hits"] += hit
            features["string_nums_total"] += 1
        elif level == "comments":
            features["comment_hits"] += hit
            features["comment_nums_total"] += 1
        elif level == "docstrings":
            features["docstring_hits"] += hit
            features["docstring_nums_total"] += 1

    # Normalize hits by totals
    for key in ["class", "function", "variable", "string", "comment", "docstring"]:
        hits_key = f"{key}_hits"
        total_key = f"{key}_nums_total"
        if features[total_key] > 0:
            features[hits_key] = features[hits_key] / features[total_key]
        else:
            features[hits_key] = 0.0
    
    logger.debug(f"Feature extraction complete. Processed {len(batch_prompts)} elements.")
    logger.debug(f"Final extracted features: {features}")
    
    return features
