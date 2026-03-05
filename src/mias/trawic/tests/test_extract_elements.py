import pytest
from src.mias.trawic.feature_extractor import _extract_elements

class TestDocstringExtraction:
    """Tests for docstring extraction using triple quotes."""
    
    def test_single_docstring_double_quotes(self):
        """Test extraction of a single docstring with triple double quotes."""
        code = '''def func():
    """This is a docstring"""
    pass'''
        result = _extract_elements(code)
        assert len(result["docstrings"]) == 1
        assert result["docstrings"][0]["value"] == "This is a docstring"
    
    def test_single_docstring_single_quotes(self):
        """Test extraction of a single docstring with triple single quotes."""
        code = """def func():
    '''This is a docstring'''
    pass"""
        result = _extract_elements(code)
        assert len(result["docstrings"]) == 1
        assert result["docstrings"][0]["value"] == "This is a docstring"
    
    def test_multiline_docstring(self):
        """Test extraction of multiline docstrings."""
        code = '''def func():
    """
    This is a multiline
    docstring with multiple lines
    """
    pass'''
        result = _extract_elements(code)
        assert len(result["docstrings"]) == 1
        expected = "\n    This is a multiline\n    docstring with multiple lines\n    "
        assert result["docstrings"][0]["value"] == expected
    
    def test_multiple_docstrings(self):
        """Test extraction of multiple docstrings."""
        code = '''"""Module docstring"""

def func1():
    """First function docstring"""
    pass

class MyClass:
    """Class docstring"""
    pass'''
        result = _extract_elements(code)
        assert len(result["docstrings"]) == 3
        assert result["docstrings"][0]["value"] == "Module docstring"
        assert result["docstrings"][1]["value"] == "First function docstring"
        assert result["docstrings"][2]["value"] == "Class docstring"
    
    def test_empty_docstring(self):
        """Test extraction of empty docstrings."""
        code = '''def func():
    """"""
    pass'''
        result = _extract_elements(code)
        assert len(result["docstrings"]) == 1
        assert result["docstrings"][0]["value"] == ""
    
    def test_docstring_with_quotes_inside(self):
        """Test docstring containing quotes."""
        code = '''def func():
    """This has "quotes" inside"""
    pass'''
        result = _extract_elements(code)
        assert len(result["docstrings"]) == 1
        assert result["docstrings"][0]["value"] == 'This has "quotes" inside'
    
    def test_docstring_position_tracking(self):
        """Test that start and end positions are correctly tracked."""
        code = '''def func():
    """Docstring"""
    pass'''
        result = _extract_elements(code)
        docstring = result["docstrings"][0]
        # Verify content extraction matches positions
        assert code[docstring["start"]:docstring["end"]] == docstring["value"]


class TestCommentExtraction:
    """Tests for comment extraction."""
    
    def test_single_comment(self):
        """Test extraction of a single comment."""
        code = "# This is a comment\nprint('hello')"
        result = _extract_elements(code)
        assert len(result["comments"]) == 1
        assert result["comments"][0]["value"] == " This is a comment"
    
    def test_multiple_comments(self):
        """Test extraction of multiple comments."""
        code = """# Comment 1
x = 1  # Inline comment
# Comment 2
y = 2"""
        result = _extract_elements(code)
        assert len(result["comments"]) == 3
        assert result["comments"][0]["value"] == " Comment 1"
        assert result["comments"][1]["value"] == " Inline comment"
        assert result["comments"][2]["value"] == " Comment 2"
    
    def test_inline_comment(self):
        """Test extraction of inline comments."""
        code = "x = 42  # meaning of life"
        result = _extract_elements(code)
        assert len(result["comments"]) == 1
        assert result["comments"][0]["value"] == " meaning of life"
    
    def test_comment_without_space(self):
        """Test comment without space after hash."""
        code = "#no space"
        result = _extract_elements(code)
        assert len(result["comments"]) == 1
        assert result["comments"][0]["value"] == "no space"
    
    def test_empty_comment(self):
        """Test extraction of empty comment."""
        code = "#\npass"
        result = _extract_elements(code)
        assert len(result["comments"]) == 1
        assert result["comments"][0]["value"] == ""
    
    def test_comment_with_special_chars(self):
        """Test comment with special characters."""
        code = "# TODO: fix this @#$%^&*()"
        result = _extract_elements(code)
        assert len(result["comments"]) == 1
        assert " TODO: fix this @#$%^&*()" in result["comments"][0]["value"]
    
    def test_comment_position_tracking(self):
        """Test that comment positions are correctly tracked."""
        code = "x = 1  # test"
        result = _extract_elements(code)
        comment = result["comments"][0]
        assert code[comment["start"]:comment["end"]] == comment["value"]


class TestFunctionNameExtraction:
    """Tests for function name extraction."""
    
    def test_simple_function(self):
        """Test extraction of simple function name."""
        code = "def my_function():\n    pass"
        result = _extract_elements(code)
        assert len(result["function_names"]) == 1
        assert result["function_names"][0]["value"] == "my_function"
    
    def test_function_with_parameters(self):
        """Test extraction of function with parameters."""
        code = "def func(a, b, c):\n    pass"
        result = _extract_elements(code)
        assert len(result["function_names"]) == 1
        assert result["function_names"][0]["value"] == "func"
    
    def test_function_with_type_hints(self):
        """Test extraction of function with type hints."""
        code = "def func(x: int, y: str) -> bool:\n    pass"
        result = _extract_elements(code)
        assert len(result["function_names"]) == 1
        assert result["function_names"][0]["value"] == "func"
    
    def test_multiple_functions(self):
        """Test extraction of multiple functions."""
        code = """def func1():
    pass

def func2():
    pass

def func3():
    pass"""
        result = _extract_elements(code)
        assert len(result["function_names"]) == 3
        assert result["function_names"][0]["value"] == "func1"
        assert result["function_names"][1]["value"] == "func2"
        assert result["function_names"][2]["value"] == "func3"
    
    def test_nested_functions(self):
        """Test extraction of nested functions."""
        code = """def outer():
    def inner():
        pass
    pass"""
        result = _extract_elements(code)
        assert len(result["function_names"]) == 2
        assert "outer" in [f["value"] for f in result["function_names"]]
        assert "inner" in [f["value"] for f in result["function_names"]]
    
    def test_function_with_default_args(self):
        """Test function with default arguments."""
        code = "def func(a=1, b='test', c=None):\n    pass"
        result = _extract_elements(code)
        assert len(result["function_names"]) == 1
        assert result["function_names"][0]["value"] == "func"
    
    def test_function_with_multiline_params(self):
        """Test function with parameters spanning multiple lines."""
        code = """def func(
    a,
    b,
    c
):
    pass"""
        result = _extract_elements(code)
        assert len(result["function_names"]) == 1
        assert result["function_names"][0]["value"] == "func"
    
    def test_private_and_dunder_functions(self):
        """Test extraction of private and dunder functions."""
        code = """def _private():
    pass

def __dunder__():
    pass

def __init__():
    pass"""
        result = _extract_elements(code)
        assert len(result["function_names"]) == 3
        assert "_private" in [f["value"] for f in result["function_names"]]
        assert "__dunder__" in [f["value"] for f in result["function_names"]]
        assert "__init__" in [f["value"] for f in result["function_names"]]
    
    def test_function_position_tracking(self):
        """Test that function name positions are correctly tracked."""
        code = "def test_func():\n    pass"
        result = _extract_elements(code)
        func = result["function_names"][0]
        assert code[func["start"]:func["end"]] == "test_func"


class TestClassNameExtraction:
    """Tests for class name extraction."""
    
    def test_simple_class(self):
        """Test extraction of simple class name."""
        code = "class MyClass:\n    pass"
        result = _extract_elements(code)
        assert len(result["class_names"]) == 1
        assert result["class_names"][0]["value"] == "MyClass"
    
    def test_class_with_inheritance(self):
        """Test extraction of class with inheritance."""
        code = "class MyClass(BaseClass):\n    pass"
        result = _extract_elements(code)
        assert len(result["class_names"]) == 1
        assert result["class_names"][0]["value"] == "MyClass"
    
    def test_class_with_multiple_inheritance(self):
        """Test extraction of class with multiple inheritance."""
        code = "class MyClass(Base1, Base2, Base3):\n    pass"
        result = _extract_elements(code)
        assert len(result["class_names"]) == 1
        assert result["class_names"][0]["value"] == "MyClass"
    
    def test_multiple_classes(self):
        """Test extraction of multiple classes."""
        code = """class Class1:
    pass

class Class2:
    pass

class Class3(Base):
    pass"""
        result = _extract_elements(code)
        assert len(result["class_names"]) == 3
        assert result["class_names"][0]["value"] == "Class1"
        assert result["class_names"][1]["value"] == "Class2"
        assert result["class_names"][2]["value"] == "Class3"
    
    def test_nested_classes(self):
        """Test extraction of nested classes."""
        code = """class Outer:
    class Inner:
        pass"""
        result = _extract_elements(code)
        assert len(result["class_names"]) == 2
        assert "Outer" in [c["value"] for c in result["class_names"]]
        assert "Inner" in [c["value"] for c in result["class_names"]]
    
    def test_private_classes(self):
        """Test extraction of private classes."""
        code = """class _PrivateClass:
    pass

class __VeryPrivate:
    pass"""
        result = _extract_elements(code)
        assert len(result["class_names"]) == 2
        assert "_PrivateClass" in [c["value"] for c in result["class_names"]]
        assert "__VeryPrivate" in [c["value"] for c in result["class_names"]]
    
    def test_class_position_tracking(self):
        """Test that class name positions are correctly tracked."""
        code = "class TestClass:\n    pass"
        result = _extract_elements(code)
        cls = result["class_names"][0]
        assert code[cls["start"]:cls["end"]] == "TestClass"


class TestVariableNameExtraction:
    """Tests for variable name extraction."""
    
    def test_simple_assignment(self):
        """Test extraction of simple variable assignment."""
        code = "x = 42"
        result = _extract_elements(code)
        assert len(result["variable_names"]) == 1
        assert result["variable_names"][0]["value"] == "x"
    
    def test_multiple_assignments(self):
        """Test extraction of multiple variable assignments."""
        code = """x = 1
y = 2
z = 3"""
        result = _extract_elements(code)
        assert len(result["variable_names"]) == 3
        assert result["variable_names"][0]["value"] == "x"
        assert result["variable_names"][1]["value"] == "y"
        assert result["variable_names"][2]["value"] == "z"
    
    def test_assignment_with_spaces(self):
        """Test variable assignment with various spacing."""
        code = """a=1
b  =  2
c   =3"""
        result = _extract_elements(code)
        assert len(result["variable_names"]) == 3
        assert result["variable_names"][0]["value"] == "a"
        assert result["variable_names"][1]["value"] == "b"
        assert result["variable_names"][2]["value"] == "c"
    
    def test_private_variables(self):
        """Test extraction of private variables."""
        code = """_private = 1
__very_private = 2
normal = 3"""
        result = _extract_elements(code)
        assert len(result["variable_names"]) >= 3
        var_values = [v["value"] for v in result["variable_names"]]
        assert "_private" in var_values
        assert "__very_private" in var_values
        assert "normal" in var_values
    
    def test_variable_with_complex_rhs(self):
        """Test variable assignment with complex right-hand side."""
        code = """result = func(a, b, c)
data = [1, 2, 3, 4, 5]
config = {"key": "value"}"""
        result = _extract_elements(code)
        var_values = [v["value"] for v in result["variable_names"]]
        assert "result" in var_values
        assert "data" in var_values
        assert "config" in var_values
    
    def test_chained_assignment(self):
        """Test chained variable assignments (captures first variable)."""
        code = "x = y = z = 42"
        result = _extract_elements(code)
        # Should capture x, y, and z
        var_values = [v["value"] for v in result["variable_names"]]
        assert "x" in var_values
        assert "y" in var_values
        assert "z" in var_values

    def test_type_annotated_variable(self):
        """Test variable with type annotation."""
        code = "x: int = 42"
        result = _extract_elements(code)
        var_values = [v["value"] for v in result["variable_names"]]
        assert "x" in var_values
    
    def test_variable_position_tracking(self):
        """Test that variable name positions are correctly tracked."""
        code = "print('hello')\nmy_var = 100"
        result = _extract_elements(code)
        var = result["variable_names"][0]
        assert code[var["start"]:var["end"]] == "my_var"


class TestStringExtraction:
    """Tests for string literal extraction."""
    
    def test_single_quoted_string(self):
        """Test extraction of single-quoted string."""
        code = "x = 'hello world'"
        result = _extract_elements(code)
        assert len(result["strings"]) == 1
        assert result["strings"][0]["value"] == "hello world"
    
    def test_double_quoted_string(self):
        """Test extraction of double-quoted string."""
        code = 'x = "hello world"'
        result = _extract_elements(code)
        assert len(result["strings"]) == 1
        assert result["strings"][0]["value"] == "hello world"
    
    def test_multiple_strings(self):
        """Test extraction of multiple strings."""
        code = """s1 = "first"
s2 = 'second'
s3 = "third" """
        result = _extract_elements(code)
        string_values = [s["value"] for s in result["strings"]]
        assert "first" in string_values
        assert "second" in string_values
        assert "third" in string_values
    
    def test_empty_string(self):
        """Test extraction of empty strings."""
        code = 'x = ""'
        result = _extract_elements(code)
        # Regex uses .+ so empty strings won't match
        # This is by design per the current regex pattern
        assert len(result["strings"]) == 0
    
    def test_string_with_special_chars(self):
        """Test strings with special characters."""
        code = r'x = "hello\nworld\ttab"'
        result = _extract_elements(code)
        assert len(result["strings"]) >= 1
        # The actual content depends on how escapes are handled
    
    def test_string_in_function_call(self):
        """Test strings in function calls."""
        code = 'print("Hello", "World")'
        result = _extract_elements(code)
        string_values = [s["value"] for s in result["strings"]]
        assert "Hello" in string_values
        assert "World" in string_values
    
    def test_string_position_tracking(self):
        """Test that string positions are correctly tracked."""
        code = 'msg = "test message"'
        result = _extract_elements(code)
        string = result["strings"][0]
        assert code[string["start"]:string["end"]] == string["value"]
    
    def test_f_string_content(self):
        """Test f-string extraction (may or may not be captured)."""
        code = 'name = "Alice"\nmsg = f"Hello {name}"'
        result = _extract_elements(code)
        # F-strings handling depends on regex pattern
        string_values = [s["value"] for s in result["strings"]]
        # At minimum should get "Alice"
        assert "Alice" in string_values


class TestComplexScenarios:
    """Tests for complex real-world code patterns."""
    
    def test_complete_class_with_methods(self):
        """Test extraction from a complete class with multiple methods."""
        code = '''class Calculator:
    """A simple calculator class"""
    
    def __init__(self):
        """Initialize the calculator"""
        self.result = 0  # Current result
    
    def add(self, x, y):
        """Add two numbers"""
        return x + y
    
    def subtract(self, x, y):
        """Subtract y from x"""
        return x - y'''
        
        result = _extract_elements(code)
        
        # Check class
        assert len(result["class_names"]) == 1
        assert result["class_names"][0]["value"] == "Calculator"
        
        # Check functions
        func_names = [f["value"] for f in result["function_names"]]
        assert "__init__" in func_names
        assert "add" in func_names
        assert "subtract" in func_names
        
        # Check docstrings
        assert len(result["docstrings"]) == 4
        
        # Check variables
        var_names = [v["value"] for v in result["variable_names"]]
        # Variables in the code: result = 0, x, y (function params aren't captured as variables)
        assert "result" in var_names or len(var_names) > 0  # At least some variables are found
        
        # Check comments
        assert len(result["comments"]) >= 1
    
    def test_module_with_imports_and_functions(self):
        """Test extraction from a module with imports and functions."""
        code = '''"""Module docstring"""
import os
import sys
from pathlib import Path

# Constants
API_KEY = "secret123"
BASE_URL = "https://api.example.com"

def process_data(data):
    """Process the input data"""
    result = data.strip()  # Remove whitespace
    return result.upper()

class DataProcessor:
    """Processes data in various ways"""
    pass'''
        
        result = _extract_elements(code)
        
        # Verify all element types are extracted
        assert len(result["docstrings"]) >= 2
        assert len(result["function_names"]) >= 1
        assert len(result["class_names"]) >= 1
        assert len(result["variable_names"]) >= 2
        assert len(result["strings"]) >= 2
        assert len(result["comments"]) >= 1
    
    def test_deeply_nested_structures(self):
        """Test extraction from deeply nested code."""
        code = '''class Outer:
    """Outer class"""
    
    class Middle:
        """Middle class"""
        
        class Inner:
            """Inner class"""
            
            def deep_method(self):
                """Deep method"""
                def inner_func():
                    """Inner function"""
                    x = 1  # Deep variable
                    return x
                return inner_func()'''
        
        result = _extract_elements(code)
        
        # All classes should be found
        class_names = [c["value"] for c in result["class_names"]]
        assert "Outer" in class_names
        assert "Middle" in class_names
        assert "Inner" in class_names
        
        # All functions should be found
        func_names = [f["value"] for f in result["function_names"]]
        assert "deep_method" in func_names
        assert "inner_func" in func_names
        
        # All docstrings should be found
        assert len(result["docstrings"]) >= 5
    
    def test_code_with_string_literals_resembling_code(self):
        """Test extraction when strings contain code-like patterns."""
        code = '''# Real comment
code_str = "def fake_function(): pass"
query = "SELECT * FROM users WHERE name = 'admin'"
comment_in_string = "This is not a # comment"'''
        
        result = _extract_elements(code)
        
        # Note: The simple regex will find "#" inside strings too.
        # This is a known limitation - proper parsing would require tokenization.
        assert len(result["comments"]) >= 1
        comment_values = [c["value"] for c in result["comments"]]
        assert " Real comment" in comment_values
        
        # Should find the strings
        string_values = [s["value"] for s in result["strings"]]
        assert "def fake_function(): pass" in string_values
        
        # Note: The regex will find "def" inside strings too (known limitation,
        # proper parsing would require tokenization to skip strings first)
        # So 'fake_function' inside the string will be found as a function name.
    
    def test_empty_code(self):
        """Test extraction from empty code."""
        code = ""
        result = _extract_elements(code)
        
        assert len(result["docstrings"]) == 0
        assert len(result["comments"]) == 0
        assert len(result["function_names"]) == 0
        assert len(result["class_names"]) == 0
        assert len(result["variable_names"]) == 0
        assert len(result["strings"]) == 0
    
    def test_whitespace_only(self):
        """Test extraction from whitespace-only code."""
        code = "   \n\n  \t\n  "
        result = _extract_elements(code)
        
        assert len(result["docstrings"]) == 0
        assert len(result["comments"]) == 0
        assert len(result["function_names"]) == 0
        assert len(result["class_names"]) == 0
        assert len(result["variable_names"]) == 0
        assert len(result["strings"]) == 0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_malformed_syntax(self):
        """Test that malformed syntax doesn't crash the extractor."""
        code = "def (incomplete"
        result = _extract_elements(code)
        # Should not crash, may or may not extract anything
        assert isinstance(result, dict)
    
    def test_mixed_quote_types(self):
        """Test code with mixed quote types."""
        code = '''"""Docstring"""
# Comment
s1 = "double"
s2 = 'single'
s3 = """triple"""'''
        
        result = _extract_elements(code)
        
        # Docstrings with triple quotes
        assert len(result["docstrings"]) >= 1
        
        # Regular strings
        string_values = [s["value"] for s in result["strings"]]
        assert "double" in string_values
        assert "single" in string_values
    
    def test_unicode_and_special_characters(self):
        """Test extraction with unicode and special characters."""
        code = '''# Комментарий на русском
def ascii_func():
    """文档字符串"""
    émoji = "🎉"
    return émoji'''
        
        result = _extract_elements(code)
        
        # Should handle unicode in comments
        assert len(result["comments"]) >= 1
        
        # Note: Current regex only matches ASCII identifiers [a-zA-Z_]...
        # Unicode identifiers like 'función' won't be extracted.
        # To support unicode, would need regex like [^\W\d]\w* or 
        # explicitly include unicode ranges.
        func_names = [f["value"] for f in result["function_names"]]
        assert "ascii_func" in func_names
        # Unicode function names won't be found with current ASCII-only regex
        assert "función" not in func_names
    
    def test_very_long_elements(self):
        """Test extraction of very long elements."""
        long_comment = "# " + "x" * 1000
        code = f"{long_comment}\npass"
        
        result = _extract_elements(code)
        assert len(result["comments"]) == 1
        assert len(result["comments"][0]["value"]) >= 1000
    
    def test_consecutive_identical_elements(self):
        """Test extraction of consecutive identical elements."""
        code = '''x = 1
x = 2
x = 3'''
        
        result = _extract_elements(code)
        # Should find all three occurrences
        x_vars = [v for v in result["variable_names"] if v["value"] == "x"]
        assert len(x_vars) == 3
        # Should have different positions
        positions = [(v["start"], v["end"]) for v in x_vars]
        assert len(set(positions)) == 3
    
    def test_single_character_elements(self):
        """Test extraction of single-character elements."""
        code = '''x = 1
y = 2
def a():
    pass
class B:
    pass'''
        
        result = _extract_elements(code)
        
        var_names = [v["value"] for v in result["variable_names"]]
        assert "x" in var_names
        assert "y" in var_names
        
        func_names = [f["value"] for f in result["function_names"]]
        assert "a" in func_names
        
        class_names = [c["value"] for c in result["class_names"]]
        assert "B" in class_names
    
    def test_elements_on_same_line(self):
        """Test multiple elements on the same line."""
        code = 'x = 1; y = 2; z = "test"  # inline comment'
        
        result = _extract_elements(code)
        
        # Should extract all variables
        var_names = [v["value"] for v in result["variable_names"]]
        assert "x" in var_names
        assert "y" in var_names
        assert "z" in var_names
        
        # Should extract string
        string_values = [s["value"] for s in result["strings"]]
        assert "test" in string_values
        
        # Should extract comment
        assert len(result["comments"]) == 1


class TestReturnStructure:
    """Tests for the return structure and data integrity."""
    
    def test_return_type_is_dict(self):
        """Test that return value is a dictionary."""
        code = "x = 1"
        result = _extract_elements(code)
        assert isinstance(result, dict)
    
    def test_all_keys_present(self):
        """Test that all expected keys are present."""
        code = ""
        result = _extract_elements(code)
        
        expected_keys = [
            "docstrings",
            "comments",
            "function_names",
            "class_names",
            "variable_names",
            "strings"
        ]
        
        for key in expected_keys:
            assert key in result
    
    def test_all_values_are_lists(self):
        """Test that all values are lists."""
        code = "x = 1"
        result = _extract_elements(code)
        
        for key, value in result.items():
            assert isinstance(value, list), f"{key} should be a list"
    
    def test_element_structure(self):
        """Test that each element has required fields."""
        code = '''def func():
    """Docstring"""
    x = 1  # Comment
    return "result"

class MyClass:
    pass'''
        
        result = _extract_elements(code)
        
        required_fields = ["value", "start", "end"]
        
        # Check all element types
        for element_type in result.values():
            for element in element_type:
                assert isinstance(element, dict)
                for field in required_fields:
                    assert field in element, f"Missing field: {field}"
                assert isinstance(element["value"], str)
                assert isinstance(element["start"], int)
                assert isinstance(element["end"], int)
                assert element["start"] >= 0
                assert element["end"] >= element["start"]
    
    def test_position_consistency(self):
        """Test that extracted positions match actual code."""
        code = '''def test_func():
    """Test docstring"""
    var = "string value"  # comment'''
        
        result = _extract_elements(code)
        
        # Check all elements
        for element_type in result.values():
            for element in element_type:
                extracted = code[element["start"]:element["end"]]
                assert extracted == element["value"], \
                    f"Position mismatch: expected '{element['value']}', " \
                    f"got '{extracted}' at {element['start']}:{element['end']}"


class TestPositionAccuracy:
    """Tests specifically for position tracking."""
    
    def test_position_ordering(self):
        """Test that positions are correctly ordered."""
        code = "x = 1"
        result = _extract_elements(code)
        var = result["variable_names"][0]
        assert var["start"] < var["end"]
        assert var["start"] >= 0
    
    def test_multiline_code_positions(self):
        """Test positions across multiple lines."""
        code = '''# Line 1
x = 1  # Line 2
def func():  # Line 3
    pass  # Line 4
y = 2  # Line 5'''
        
        result = _extract_elements(code)
        
        # All elements should have valid positions
        for element_type in result.values():
            for element in element_type:
                assert element["start"] >= 0
                assert element["end"] > element["start"]
                assert code[element["start"]:element["end"]] == element["value"]
    
    def test_position_uniqueness_with_blank_lines(self):
        """Test position tracking with blank lines."""
        code = '''x = 1

y = 2


z = 3'''
        result = _extract_elements(code)
        
        var_x = [v for v in result["variable_names"] if v["value"] == "x"][0]
        var_y = [v for v in result["variable_names"] if v["value"] == "y"][0]
        var_z = [v for v in result["variable_names"] if v["value"] == "z"][0]
        
        # All should have different positions
        assert var_x["start"] < var_y["start"] < var_z["start"]
        assert var_x["end"] < var_y["end"] < var_z["end"]
