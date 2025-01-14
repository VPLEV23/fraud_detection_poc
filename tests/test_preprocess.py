import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess_text

def test_preprocess_text():
    input_text = "  Hello, World! This is a test.  "
    expected_output = "hello world this is a test"
    assert preprocess_text(input_text) == expected_output
