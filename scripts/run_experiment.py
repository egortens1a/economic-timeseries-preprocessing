"""Запуск CLI из исходников. Также можно использовать python -m tsprep.cli."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from tsprep.cli import main

if __name__ == "__main__":
    main()
