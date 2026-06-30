"""Config loading with OmegaConf: base.yaml + override merge.

Convention: a leaf config may declare `_base_: <relative path>` (string or list)
pointing at one or more parent YAMLs. Parents are merged first (in order), then the
child overrides them. This gives the "base + override" composition asked for in
CLAUDE.md without any framework magic — just OmegaConf.merge.
"""
import os

from omegaconf import OmegaConf


def _load_with_bases(path, _seen=None):
    _seen = _seen or set()
    path = os.path.abspath(path)
    if path in _seen:
        raise ValueError(f"circular _base_ include at {path}")
    _seen.add(path)

    cfg = OmegaConf.load(path)
    bases = cfg.pop("_base_", None)
    if bases is None:
        return cfg
    if isinstance(bases, str):
        bases = [bases]

    here = os.path.dirname(path)
    merged = OmegaConf.create({})
    for b in bases:
        merged = OmegaConf.merge(merged, _load_with_bases(os.path.join(here, b), _seen))
    return OmegaConf.merge(merged, cfg)


def load_config(path, cli_overrides=None):
    """Load a config file, resolving `_base_` includes, then apply CLI dotlist overrides.

    cli_overrides: list like ["train.epochs=2", "model.backbone.name=rtformer"].
    """
    cfg = _load_with_bases(path)
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(cli_overrides)))
    return cfg
