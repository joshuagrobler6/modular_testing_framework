from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules

_LOADED = False


def load_all_nodes() -> None:
    global _LOADED
    if _LOADED:
        return

    for module_info in iter_modules(__path__, prefix=f"{__name__}."):
        if module_info.ispkg:
            continue
        import_module(module_info.name)

    _LOADED = True


__all__ = ["load_all_nodes"]
