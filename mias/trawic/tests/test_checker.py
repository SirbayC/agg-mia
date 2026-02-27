import unittest
import os
import sys

# Ensure we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.checker.checker import Checker

class TestChecker(unittest.TestCase):
    def setUp(self):
        """
        Creates a temporary dummy python file with known content
        so we can assert EXACTLY what should be found.
        """
        self.test_file_path = os.path.join(os.getcwd(), 'dummy_test_script.py')
        
        # This content is designed to trigger all regex patterns in checker.py
        self.dummy_content = '''"""Module docstring."""

# A sample comment
GLOBAL_VAR = "some_string"

class MyClass:
    pass

def my_function(arg1):
    return arg1
'''
        with open(self.test_file_path, "w") as f:
            f.write(self.dummy_content)

        self.checker = Checker(self.test_file_path)

    def tearDown(self):
        """Clean up the dummy file after tests run."""
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

    def test_prepare_input_correctness(self):
        """
        Test that prepare_input actually extracts the correct names and values.
        """
        self.checker.prepare_input()
        data = self.checker.processed_input

        # 1. Check Variable Names (Should find 'GLOBAL_VAR')
        # values() returns the extracted names. We convert to list to check content.
        extracted_vars = list(data['variable_names'].values())
        self.assertIn('GLOBAL_VAR', extracted_vars, "Failed to extract variable 'GLOBAL_VAR'")

        # 2. Check Strings (Should find '"some_string"')
        extracted_strings = list(data['strings'].values())
        self.assertIn('"some_string"', extracted_strings, "Failed to extract string literal")

        # 3. Check Function Names (Should find 'my_function')
        # Checker stores functions as tuples: (name, args)
        extracted_funcs = list(data['function_names'].values())
        self.assertIn(('my_function', 'arg1'), extracted_funcs, "Failed to extract function 'my_function'")

        # 4. Check Comments
        # Note: Checker regex r"\s*#(.*)" captures the text AFTER the #
        extracted_comments = list(data['comments'].values())
        self.assertTrue(any(' A sample comment' in c for c in extracted_comments), "Failed to extract comment")

    def test_separate_script_valid(self):
        """
        Test splitting the script using a word that exists in the file.
        """
        # We know "GLOBAL_VAR" is in the file.
        target_word = "GLOBAL_VAR"
        
        # We need the line number. In our dummy string, GLOBAL_VAR is on line 4.
        line_num = 4 

        prefix, suffix = self.checker.separate_script(
            self.checker.original_input, target_word, line_num
        )

        # 1. Define exactly what we expect
        # Note: The function joins lines with "\n" but does NOT add a trailing newline to the prefix
        # even though "GLOBAL_VAR" is on the next line.
        expected_prefix = '"""Module docstring."""\n\n# A sample comment'
        
        # Suffix is everything after "GLOBAL_VAR".
        # It preserves the rest of the line (' = "some_string"') and the subsequent lines.
        expected_suffix = ' = "some_string"\n\nclass MyClass:\n    pass\n\ndef my_function(arg1):\n    return arg1\n'

        # 2. Assert Exact Match
        self.assertEqual(prefix, expected_prefix, "Prefix content did not match exactly")
        self.assertEqual(suffix, expected_suffix, "Suffix content did not match exactly")

    def test_prepare_inputs_for_infill_structure(self):
        """
        Test that the generated candidates have the correct dictionary structure.
        """
        # We check 'variable_names' because we know we have one.
        candidates = self.checker.prepare_inputs_for_infill("variable_names")
        
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0, "Should have found at least one variable candidate")
        
        first_candidate = candidates[0]
        required_keys = ["infill", "prefix", "suffix", "level"]
        
        for key in required_keys:
            self.assertIn(key, first_candidate, f"Candidate missing key: {key}")
        
        self.assertEqual(first_candidate['infill'], "GLOBAL_VAR")
        self.assertEqual(first_candidate['level'], "variable_names")

    def test_check_similarity_logic(self):
        """
        Test the static method with specific Scenarios (Exact vs Fuzzy).
        """
        candidate = {"infill": "secret_code"}

        # 1. Exact Match - Success
        score = self.checker.check_similarity("secret_code", candidate, "exact")
        self.assertEqual(score, 1, "Exact match should return 1")

        # 2. Exact Match - Failure
        score = self.checker.check_similarity("wrong_code", candidate, "exact")
        self.assertEqual(score, 0, "Exact mismatch should return 0")

        # 3. Fuzzy Match
        score = self.checker.check_similarity("secret_c0de", candidate, "fuzzy")
        self.assertGreater(score, 90, "Fuzzy match should be > 90 for similar strings (secret_c0de vs secret_code)")
        
        # 4. Check for None (Model failure handling)
        score = self.checker.check_similarity(None, candidate, "exact")
        self.assertEqual(score, 0, "None output should result in 0 score")

if __name__ == "__main__":
    unittest.main()