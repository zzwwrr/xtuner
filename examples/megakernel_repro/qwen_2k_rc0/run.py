"""Repository-local reproduction entry; no GPU launch without --execute."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'common'))
from run_training import main

if __name__ == '__main__':
    main(HERE.name, HERE)
