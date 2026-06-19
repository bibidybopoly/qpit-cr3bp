import sys
import os

# Add src/ to the Python path so tests can import modules from it.
# Runs automatically before any tests, regardless of where pytest is invoked.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))