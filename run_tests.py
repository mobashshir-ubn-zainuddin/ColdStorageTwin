import unittest
import glob
import os

def run_all_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Find all test files
    test_files = glob.glob('tests/test_*.py')
    for file in test_files:
        module_name = os.path.basename(file).replace('.py', '')
        # We need to import them. Since they used pytest, let's adapt the functions to TestCases.
        # Actually, let's just run them as functions using a simple loop for now.
        print(f"Running {file}...")
        try:
            # This is a hacky way to run pytest-style functions with unittest
            # Better: rewrite them.
            pass
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    run_all_tests()
