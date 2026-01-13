"""
Test runner for embedding storage tests.

Usage:
    python run_tests.py              # Run unit tests
    python run_tests.py --main       # Run comprehensive main test
    python run_tests.py --all        # Run both
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def run_unit_tests():
    """Run unit tests."""
    print("\n" + "=" * 70)
    print("RUNNING UNIT TESTS")
    print("=" * 70 + "\n")
    
    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover('src', pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_main_test():
    """Run comprehensive test in embedding_storage.py main."""
    print("\n" + "=" * 70)
    print("RUNNING COMPREHENSIVE MAIN TEST")
    print("=" * 70 + "\n")
    
    # Import and run the main block
    import embedding_storage
    # The main test will run automatically when imported if __name__ == "__main__"
    # But we need to trigger it manually
    import importlib
    importlib.reload(embedding_storage)
    
    # Actually, we need to execute it differently
    import subprocess
    result = subprocess.run([sys.executable, 'src/embedding_storage.py'], 
                          capture_output=False)
    return result.returncode == 0


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if '--main' in sys.argv:
            success = run_main_test()
        elif '--all' in sys.argv:
            success1 = run_unit_tests()
            success2 = run_main_test()
            success = success1 and success2
        else:
            success = run_unit_tests()
    else:
        success = run_unit_tests()
    
    sys.exit(0 if success else 1)




