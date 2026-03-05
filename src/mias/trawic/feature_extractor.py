"""
TraWiC Feature Extraction Pipeline

Extracts membership inference features by testing if a model can reconstruct
masked code elements (function names, variables, comments, etc.).
"""

import logging
import re
from typing import Dict, Tuple

import torch
from fuzzywuzzy import fuzz

logger = logging.getLogger(__name__)


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
            line_num = code.count("\n", 0, match.start()) + 1
            elements["docstrings"].append({
                "value": match.group(2),  # Only the content, not the quotes
                "start": match.start(2),  # Position of content start
                "end": match.end(2),      # Position of content end
                "line": line_num
            })
        
        # Extract comments (content only, after '#')
        for match in re.finditer(r"\s*#(.*)", code):
            line_num = code.count("\n", 0, match.start()) + 1
            comment_start = match.start(1)
            comment_end = match.end(1)
            elements["comments"].append({
                "value": code[comment_start:comment_end],
                "start": comment_start,
                "end": comment_end,
                "line": line_num
            })
        
        # Extract function names
        for match in re.finditer(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)", code):
            line_num = code.count("\n", 0, match.start()) + 1
            elements["function_names"].append({
                "value": match.group(1),
                "start": match.start(1),
                "end": match.end(1),
                "line": line_num
            })
        
        # Extract class names
        for match in re.finditer(r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\((.*?)\))?", code):
            line_num = code.count("\n", 0, match.start()) + 1
            elements["class_names"].append({
                "value": match.group(1),
                "start": match.start(1),
                "end": match.end(1),
                "line": line_num
            })
        
        # Extract variable names (assignments)
        for match in re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=", code):
            line_num = code.count("\n", 0, match.start()) + 1
            elements["variable_names"].append({
                "value": match.group(1),
                "start": match.start(1),
                "end": match.end(1),
                "line": line_num
            })
        
        # Extract strings (only the content inside single/double quotes)
        for match in re.finditer(r'(["\'])(.+?)\1', code):
            line_num = code.count("\n", 0, match.start()) + 1
            elements["strings"].append({
                "value": match.group(2),  # Only the content, not the quotes
                "start": match.start(2),  # Position of content start
                "end": match.end(2),      # Position of content end
                "line": line_num
            })
    except Exception as e:
        logger.warning(f"Error extracting elements: {e}")
    
    return elements


def _create_infill_prompt(code: str, element: Dict, level: str, fim_tokens: Dict[str, str]) -> Tuple[str, str]:
    """
    Create FIM prompt for infilling by masking the element.
    
    Uses character indices only: delimiters and punctuation are naturally preserved
    in prefix/suffix by slicing the original code.
    
    Constructs StarCoder2 canonical FIM format:
    <fim_prefix>[code_before_gap]<fim_suffix>[code_after_gap]<fim_middle>
    
    Args:
        code: Full code string
        element: Element dict with start, end, value
        level: Type of element
        fim_tokens: Dict with FIM token strings
        
    Returns:
        Tuple of (infill_target, fim_prompt)
    """
    prefix = code[:element["start"]]
    suffix = code[element["end"]:]
    
    prompt = f"{fim_tokens['prefix']}{prefix}{fim_tokens['suffix']}{suffix}{fim_tokens['middle']}"
    
    return element["value"], prompt


def _run_infill(
    model,
    tokenizer,
    prompt: str,
    device: str,
    fim_tokens: Dict[str, str],
    max_tokens: int = 50
) -> str:
    """
    Run model infilling using FIM prompt.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Complete FIM prompt string
        device: Device to run on ('cuda' or 'cpu')
        fim_tokens: Dict with FIM token strings (for parsing output)
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated infill text
    """
    try:
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - max_tokens
        ).to(device)
        
        # Check if input is too long
        if inputs.input_ids.shape[1] + max_tokens > 2048:
            return "too_many_tokens"
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.2,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract generated infill after <fim_middle>
        if fim_tokens['middle'] in full_output:
            start_idx = full_output.find(fim_tokens['middle']) + len(fim_tokens['middle'])
            end_idx = full_output.find(fim_tokens['endoftext'], start_idx)
            if end_idx == -1:
                end_idx = len(full_output)
            return full_output[start_idx:end_idx].strip()
        
        return None
        
    except Exception as e:
        logger.warning(f"Error during infill: {e}")
        return None


def _check_similarity(target: str, output: str, metric: str = "exact", threshold: int = 100) -> int:
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
    
    if metric == "exact":
        return 1 if target.strip().upper() == output.strip().upper() else 0
    else:  # fuzzy
        similarity = fuzz.ratio(target.strip(), output.strip())
        return 1 if similarity >= threshold else 0


def extract_features(
    code: str,
    model,
    tokenizer,
    device: str = "cuda",
    syntactic_threshold: int = 100,
    semantic_threshold: int = 20
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
        syntactic_threshold: Threshold for exact syntax matches (default 100 = perfect match)
        semantic_threshold: Threshold for fuzzy semantic matches (default 20)
        
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
    # FIM tokens for StarCoder models
    fim_tokens = {
        'prefix': '<fim_prefix>',
        'middle': '<fim_middle>',
        'suffix': '<fim_suffix>',
        'pad': '<fim_pad>',
        'endoftext': '<|endoftext|>'
    }
    
    features = {
        "class_hits": 0,
        "class_nums_total": 0,
        "function_hits": 0,
        "function_nums_total": 0,
        "variable_hits": 0,
        "variable_nums_total": 0,
        "string_hits": 0,
        "string_nums_total": 0,
        "comment_hits": 0,
        "comment_nums_total": 0,
        "docstring_hits": 0,
        "docstring_nums_total": 0,
    }
    
    # Extract all elements
    elements = _extract_elements(code)
    
    # Process each element type
    for level, element_list in elements.items():
        # Determine similarity metric (exact for syntax, fuzzy for semantic)
        metric = "exact" if level in ["function_names", "class_names", "variable_names"] else "fuzzy"
        threshold = syntactic_threshold if metric == "exact" else semantic_threshold
        
        for element in element_list:
            # Create infill prompt
            target, prompt = _create_infill_prompt(code, element, level, fim_tokens)
            
            # Skip empty or very short elements
            if not target or len(target.strip()) < 2:
                continue
            
            # Run model infill
            output = _run_infill(model, tokenizer, prompt, device, fim_tokens)
            
            # Check similarity
            hit = _check_similarity(target, output, metric, threshold)
            
            # Update features based on element type
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
    
    return features
