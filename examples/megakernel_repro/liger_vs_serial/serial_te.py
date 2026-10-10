"""Import TE GroupedLinear without the image's broken optional FA3 package.

TE 2.11 eagerly imports attention even for MoE. This image advertises FA3
metadata but lacks flash_attn_3.flash_attn_interface. Only hide that optional
distribution during this import; do not modify shared packages or GEMM code.
"""
import importlib.metadata
import importlib.util
import warnings
from unittest.mock import patch


def load_grouped_linear():
    try:
        present = importlib.util.find_spec("flash_attn_3.flash_attn_interface") is not None
    except ModuleNotFoundError:
        present = False
    original = importlib.metadata.version

    def version(name):
        if name.replace("_", "-") == "flash-attn-3" and not present:
            raise importlib.metadata.PackageNotFoundError(name)
        return original(name)

    if not present:
        warnings.warn("Serial: disabling unavailable optional FA3 during TE import")
    with patch.object(importlib.metadata, "version", version):
        from transformer_engine.pytorch import GroupedLinear
    return GroupedLinear
