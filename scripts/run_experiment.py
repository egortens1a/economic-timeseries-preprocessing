"""Compatibility wrapper. Prefer python -m tsprep.cli or tsprep-experiment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from tsprep.cli import main

if __name__ == "__main__":
    main()
