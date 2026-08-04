"""Ensures the project root is importable (so `from src...` works in tests)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
