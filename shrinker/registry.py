from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

# Order in which kinds are applied when methods are combined.
KINDS = ("pruning", "factorization", "fusion", "quantization", "other")

REGISTRY: Dict[str, "MethodInfo"] = {}

# Method files that failed to import: {module: error}. CI fails if not empty.
LOAD_ERRORS: Dict[str, str] = {}


class UnknownMethodError(ValueError):
    pass


@dataclass(frozen=True)
class MethodInfo:
    name: str
    fn: Callable
    kind: str
    paper: Optional[str]

    @property
    def module(self) -> str:
        return self.fn.__module__

    @property
    def summary(self) -> str:
        doc = (self.fn.__doc__ or "").strip()
        return doc.splitlines()[0] if doc else ""


def register(name=None, kind: str = "other", paper: Optional[str] = None):
    """Register a compression method. Use as ``@register``; the method name is the file name."""
    if callable(name):  # bare @register
        return register()(name)
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise ValueError(f"Method name must be a non-empty string, got {name!r}")

    def wrap(fn: Callable) -> Callable:
        method_name = name if name is not None else fn.__module__.rsplit(".", 1)[-1]
        method_kind = kind
        if method_kind not in KINDS:
            logger.warning(
                "Method %r: unknown kind %r, treating it as 'other'. Known kinds: %s", method_name, kind, KINDS
            )
            method_kind = "other"
        existing = REGISTRY.get(method_name)
        # Re-registering the same function (e.g. on reload) is fine.
        if existing is not None and (existing.module, existing.fn.__qualname__) != (
            fn.__module__,
            fn.__qualname__,
        ):
            raise ValueError(
                f"Two files register the method name {method_name!r}: {existing.module} and "
                f"{fn.__module__}. Each method needs its own name."
            )
        REGISTRY[method_name] = MethodInfo(name=method_name, fn=fn, kind=method_kind, paper=paper)
        return fn

    return wrap


def get_method(name: str) -> MethodInfo:
    try:
        return REGISTRY[name]
    except KeyError:
        available = ", ".join(list_methods()) or "(none)"
        raise UnknownMethodError(
            f"Unknown method {name!r}. Available methods: {available}"
        ) from None


def list_methods(kind: Optional[str] = None) -> List[str]:
    return sorted(n for n, info in REGISTRY.items() if kind is None or info.kind == kind)


def describe_methods() -> str:
    rows = [(i.name, i.kind, i.summary) for i in (REGISTRY[n] for n in list_methods())]
    if not rows:
        return "(no methods registered)"
    w0 = max(len("method"), *(len(r[0]) for r in rows))
    w1 = max(len("kind"), *(len(r[1]) for r in rows))
    lines = [f"{'method':<{w0}}  {'kind':<{w1}}  description"]
    lines += [f"{a:<{w0}}  {b:<{w1}}  {c}" for a, b, c in rows]
    return "\n".join(lines)


def discover(package: str, path: Iterable[str]) -> None:
    """Import every module in the package (except ``_*.py``) so its ``@register`` runs."""
    for mod in sorted(pkgutil.iter_modules(path), key=lambda m: m.name):
        if mod.name.startswith("_"):
            continue
        full = f"{package}.{mod.name}"
        try:
            importlib.import_module(full)
            LOAD_ERRORS.pop(full, None)
        except Exception as e:  # noqa: BLE001  one broken file must not break the library
            LOAD_ERRORS[full] = f"{type(e).__name__}: {e}"
            logger.warning("Could not load compression method %s: %s", full, LOAD_ERRORS[full])
