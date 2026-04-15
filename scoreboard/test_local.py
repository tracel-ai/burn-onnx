#!/usr/bin/env python3
"""Local test runner for burn-onnx ONNX backend.

Usage:
    # Run all node tests (CPU only):
    python test_local.py

    # Run a specific test:
    python test_local.py -k test_abs

    # List available tests:
    python test_local.py --collect-only
"""

import os
import sys
import unittest

# Ensure backend module is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backend  # noqa: E402
import onnx.backend.test  # noqa: E402

# Create the test suite
backend_test = onnx.backend.test.BackendTest(backend, __name__)

# Exclude CUDA tests (CPU only)
backend_test.exclude(r".*_cuda")

# Populate test cases into this module's globals
globals().update(backend_test.test_cases)

if __name__ == "__main__":
    unittest.main()
