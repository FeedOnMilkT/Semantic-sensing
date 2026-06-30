"""Minimal stdout logger. No external deps; WandB/TensorBoard can wrap this later."""
import logging
import sys

_CONFIGURED = False


def get_logger(name="demo1"):
    global _CONFIGURED
    if not _CONFIGURED:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"))
        root = logging.getLogger("demo1")
        root.addHandler(h)
        root.setLevel(logging.INFO)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger("demo1").getChild(name) if name != "demo1" else logging.getLogger("demo1")
