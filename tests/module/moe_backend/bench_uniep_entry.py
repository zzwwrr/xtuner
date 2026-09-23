"""UniEP bench entry: register the patched Triton plugin before XTuner imports it.

XTuner pulls in the image Triton during import. If that happens before
``triton_dist``, the distributed dialect stays unregistered and the first
UniEP kernel aborts in LLVM. This wrapper only exists to fix that order.
"""
import runpy
import sys

import triton_dist  # noqa: F401

sys.argv[0] = "bench_moe_layer.py"
runpy.run_path(
    "tests/module/moe_backend/bench_moe_layer.py",
    run_name="__main__",
)
