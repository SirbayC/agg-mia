import re
import random
import secrets
import string

# ---------- Utility Functions ----------

def generate_random_constant():
    """generates a random constant: number or string."""
    if random.choice([True, False]):
        return str(random.randint(0, 9999))
    else:
        char_pool = string.ascii_letters + string.digits + string.punctuation
        rng = secrets.SystemRandom()
        length = 12
        return f"'{''.join(rng.choice(char_pool) for _ in range(length))}'"


def find_variables(code):
    """Regular expressions match Python variable names."""
    pattern = r"^(?!(False|None|True|and|as|assert|async|await|break|class|continue|def|del|elif|else|except|finally|for|from|global|if|import|in|is|lambda|nonlocal|not|or|pass|raise|return|try|while|with|yield)$)(?!^[A-Z])[_a-z][_a-zA-Z0-9]*$"
    tokens = re.findall(r"\b[_a-zA-Z][_a-zA-Z0-9]*\b", code)
    return [tok for tok in set(tokens) if re.match(pattern, tok)]


def find_methods(code):
    """Regular expressions match Python method names."""
    method_pattern = re.compile(r'''^\s*def\s+
        (?P<name>(?!
            (False|None|True|and|as|assert|async|await|break|class|continue|
            def|del|elif|else|except|finally|for|from|global|if|import|in|is|
            lambda|nonlocal|not|or|pass|raise|return|try|while|with|yield)\b
        )(?!^[A-Z])[_a-zA-Z][_a-zA-Z0-9]*)''', re.VERBOSE | re.MULTILINE)
    return [m.group('name') for m in method_pattern.finditer(code)]

# ---------- Perturbation Variants ----------

def insert_false_if_fixed(code):
    """Insert a fixed length useless if statement fragment."""
    lines = code.splitlines()
    idx = random.randint(0, len(lines))
    cond = random.choice(["if \"key\" != \"key\":", "if False:"])
    body = random.choice(["    void_array = [''] * 50", "    void_array[10] = 'A'"])
    new_lines = lines[:idx] + [cond] + [body] + lines[idx:]
    return '\n'.join(new_lines)


def insert_false_if_random(code):
    """Randomly extract a line from the original code as the if body."""
    lines = code.splitlines()
    idx = random.randint(0, len(lines))
    cond = random.choice(["if \"key\" != \"key\":", "if False:"])
    candidates = [l.strip() for l in lines if l.strip() and not l.strip().endswith(':')]
    stmt = random.choice(candidates) if candidates else 'pass'
    body = f"    {stmt}"
    new_lines = lines[:idx] + [cond] + [body] + lines[idx:]
    return '\n'.join(new_lines)


def insert_unused_var_existing(code):
    """Insert unused variable declarations initialized with existing variables."""
    vars = find_variables(code)
    val = random.choice(vars) if vars else generate_random_constant()
    new_var = f"unused_{random.randint(1000,9999)}"
    decl = f"{new_var} = {val}"
    # return decl + '\n' + code

    lines = code.splitlines()
    idx = next((i for i, l in enumerate(lines) if re.search(rf"\b{re.escape(val)}\b", l)), None)
    new_lines = lines[:idx] + [decl] + lines[idx:]
    return '\n'.join(new_lines)



def insert_unused_var_random(code):
    """Insert unused variable declarations initialized with random constants."""
    val = generate_random_constant()
    new_var = f"unused_{random.randint(1000,9999)}"
    decl = f"{new_var} = {val}"
    # return decl + '\n' + code

    lines = code.splitlines()
    idx = random.randint(0, len(lines))
    new_lines = lines[:idx] + [decl] + lines[idx:]
    return '\n'.join(new_lines)



def rename_var(code):
    """Randomly rename a variable."""
    vars = find_variables(code)
    if vars == []:
        return code
    name = random.choice(vars)
    prefix = 'var_'
    new_name = prefix + str(random.randint(1000,9999))
    return re.sub(rf"\b{name}\b", new_name, code)


def rename_method(code):
    """Randomly rename a method."""
    methods = find_methods(code)
    if methods == []:
        return code
    name = random.choice(methods)
    prefix = 'method_'
    new_name = prefix + str(random.randint(1000,9999))
    return re.sub(rf"\b{name}\b", new_name, code)


def insert_print_enter(code):
    """Insert Debug at the beginning of the method body to enter the printing of the method."""
    lines = code.splitlines()
    methods = find_methods(code)
    name = random.choice(methods) if methods else 'foo'
    stmt = f'print("Debug: Entering method {name}()")'

    # Insert to the next line of method definition:
    idx = next((i for i, l in enumerate(lines) if re.search(rf"\b{name}\b", l)), None)

    if idx is None:
        # If can't find it at all, insert it at the beginning of the file
        idx = 0
        indent = ''
    else:
        # Calculate indentation based on that line
        matches = re.match(r"^\s*", lines[idx])
        if matches == None:
            return code
        else:
            indent = matches.group(0)
        # indent = re.match(r"^\\s*", lines[idx]).group(0)

    # Insert after this line
    lines.insert(idx + 1, indent + stmt)
    return "\n".join(lines)


def insert_print_variable(code):
    """Insert Debug to print the value of the variable after the variable definition."""
    lines = code.splitlines()
    # If there are no lines in the original code, return directly
    if not lines:
        return code

    vars = find_variables(code)
    name = random.choice(vars) if vars else 'x'
    stmt = f'print("Debug: Variable {name} = ", {name})'

    # Find the first row containing the variable name
    # Please note to replace 'default' with 'None', as this will not directly result in 'len (lines) -1'    idx = next((i for i, l in enumerate(lines) if re.search(rf"\b{name}\b", l)), None)
    idx = next((i for i, l in enumerate(lines) if re.search(rf"\b{name}\b", l)), None)
    
    if idx is None:
        idx = 0
        indent = ''
    else:
        matches = re.match(r"^\s*", lines[idx])
        if matches == None:
            return code
        else:
            indent = matches.group(0)
        # indent = re.match(r"^\\s*", lines[idx]).group(0)

    lines.insert(idx + 1, indent + stmt)
    return "\n".join(lines)


def insert_false_loop_for(code):
    """Insert a for loop that will never execute."""
    lines = code.splitlines()
    idx = random.randint(0, len(lines))
    x, y = sorted([random.randint(0,9999) for _ in range(2)], reverse=True)
    loop = f"for _ in range({x}, {y}):"
    body = "    print(\"Debug: Entering loop\")" if random.choice([True, False]) else "    pass"
    new_lines = lines[:idx] + [loop, body] + lines[idx:]
    return '\n'.join(new_lines)


def insert_false_loop_while(code):
    """Insert a while loop that will never execute."""
    lines = code.splitlines()
    idx = random.randint(0, len(lines))
    cond = random.choice(["while False:", "while \"key\" != \"key\":"])
    body = "    print(\"Debug: Entering loop\")" if random.choice([True, False]) else "    pass"
    new_lines = lines[:idx] + [cond, body] + lines[idx:]
    return '\n'.join(new_lines)


def insert_unused_var_loop(code):
    """Combination: Insert useless loops first, then insert unused variable declarations."""
    # return insert_unused_var_existing(insert_false_loop_for(code))
    lines = code.splitlines()
    idx = random.randint(0, len(lines))
    x, y = sorted([random.randint(0,9999) for _ in range(2)], reverse=True)

    loop = f"for _ in range({x}, {y}):"
    val = generate_random_constant()
    new_var = f"unused_{random.randint(1000,9999)}"
    body = f"{new_var} = {val}"
    new_lines = lines[:idx] + [loop, body] + lines[idx:]
    return '\n'.join(new_lines)


PERTURBATIONS = {
    'false_if_fixed': insert_false_if_fixed,
    'false_if_random': insert_false_if_random,
    'unused_var_existing': insert_unused_var_existing,
    'unused_var_random': insert_unused_var_random,
    'rename_var': rename_var,
    'rename_method': rename_method,
    'print_enter': insert_print_enter,
    'print_variable': insert_print_variable,
    'false_loop_for': insert_false_loop_for,
    'false_loop_while': insert_false_loop_while,
    'unused_var_loop': insert_unused_var_loop
}

def traverse_all_variants(code):
    """Traverse all specific variables and return a dictionary:{variant_name: perturbed_code}."""
    results = {}
    for name, func in PERTURBATIONS.items():
        results[name] = func(code)
    return results
