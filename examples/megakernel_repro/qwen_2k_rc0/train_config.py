"""Fixed workload for this experiment; configuration assembly is shared."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'common'))
from qwen_config import build

trainer = build()
