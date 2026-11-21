"""Resolve Hydra-style CLI overrides into dataclasses and save them as tagged JSON.

Authoring stays in Python. A run file records types explicitly, so slices, tuples,
and dataclass trees survive a round trip. Callables are stored as import paths.
"""

import copy
import dataclasses
import importlib
import json
import sys
import types
from collections.abc import Callable, Sequence
from functools import wraps
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

_MISSING = dataclasses.MISSING


class ConfigStore:
    """Named presets addressed as ``group/name``, matching a Hydra ConfigStore."""

    def __init__(self):
        self._nodes: dict[tuple[str, str], Any] = {}

    def store(self, name: str, node: Any, *, group: str) -> None:
        key = (group, name)
        if key in self._nodes:
            raise ValueError(f"Config {group}/{name} is already registered")
        self._nodes[key] = node

    def get(self, group: str, name: str) -> Any:
        try:
            node = self._nodes[(group, name)]
        except KeyError:
            known = self.names(group)
            raise KeyError(f"Unknown {group} config {name!r}. Known: {known}") from None
        if isinstance(node, type):
            return node()
        return copy.deepcopy(node)

    def names(self, group: str) -> list[str]:
        return sorted(name for item_group, name in self._nodes if item_group == group)

    def groups(self) -> list[str]:
        return sorted({group for group, _name in self._nodes})

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self._nodes


cs = ConfigStore()


def save_config(path: str | Path, config: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dump(config), indent=2) + "\n")


def load_config(path: str | Path) -> Any:
    return load(json.loads(Path(path).read_text()))


def dump(value: Any) -> Any:
    """Encode a config tree. ``init=False`` fields are omitted."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, slice):
        return {"__type__": "slice", "value": [value.start, value.stop, value.step]}
    if isinstance(value, tuple):
        return {"__type__": "tuple", "value": [dump(item) for item in value]}
    if isinstance(value, list):
        return [dump(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Config dict keys must be strings")
        return {"__type__": "dict", "value": {key: dump(item) for key, item in value.items()}}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {
            field.name: dump(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if field.init
        }
        return {
            "__type__": _qualname(type(value)),
            "fields": fields,
        }
    if callable(value):
        path = _qualname(value)
        if "<" in path:
            raise TypeError(f"Refusing to save unimportable callable {value!r}")
        return {"__type__": "callable", "value": path}
    raise TypeError(f"Cannot save config value of type {type(value).__name__}")


def load(value: Any) -> Any:
    if isinstance(value, list):
        return [load(item) for item in value]
    if isinstance(value, dict) and "__type__" in value:
        kind = value["__type__"]
        if kind == "slice":
            return slice(*(load(item) for item in value["value"]))
        if kind == "tuple":
            return tuple(load(item) for item in value["value"])
        if kind == "dict":
            return {key: load(item) for key, item in value["value"].items()}
        if kind == "callable":
            return _import(value["value"])
        return _import(kind)(**{key: load(item) for key, item in value["fields"].items()})
    return value


def resolve(config_cls: type, argv: Sequence[str]) -> Any:
    """Build ``config_cls`` from ``key=value`` and ``group=preset`` arguments."""
    assignments = [_split_assignment(token) for token in argv]
    hints = get_type_hints(config_cls)
    selected: dict[str, Any] = {}
    consumed: set[str] = set()
    for key, raw in assignments:
        if "." in key:
            continue
        hint = hints.get(key)
        if hint is not None and dataclasses.is_dataclass(hint) and (key, raw) in cs:
            selected[key] = cs.get(key, raw)
            consumed.add(key)
    missing = [
        field.name
        for field in dataclasses.fields(config_cls)
        if field.init
        and field.default is _MISSING
        and field.default_factory is _MISSING
        and field.name not in selected
    ]
    if missing:
        details = []
        for name in missing:
            options = cs.names(name)
            if options:
                details.append(f"{name} (choose from {', '.join(options)})")
            else:
                details.append(name)
        raise ValueError(f"Missing config values: {'; '.join(details)}")
    config = config_cls(**selected)
    for key, raw in assignments:
        if key in consumed:
            continue
        config = _assign(config, key.split("."), raw)
    return config


def main(config_cls: type) -> Callable:
    """Call ``fn(config)`` after resolving CLI overrides.

    ``config_out=path`` is removed from the overrides and, when present, the
    resolved dataclass is written there as tagged JSON.
    """

    def decorate(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(argv: Sequence[str] | None = None):
            tokens = list(sys.argv[1:] if argv is None else argv)
            config_out = None
            overrides = []
            for token in tokens:
                if token.startswith("config_out="):
                    config_out = token.split("=", 1)[1]
                else:
                    overrides.append(token)
            config = resolve(config_cls, overrides)
            if config_out:
                save_config(config_out, config)
            return fn(config)

        return wrapper

    return decorate


def _assign(obj: Any, parts: list[str], raw: str) -> Any:
    head, *rest = parts
    if isinstance(obj, dict):
        if head not in obj:
            raise KeyError(f"Unknown config key {head!r}")
        updated = dict(obj)
        updated[head] = _assign(obj[head], rest, raw) if rest else _parse_like(obj[head], raw)
        return updated
    if not dataclasses.is_dataclass(obj):
        raise KeyError(f"Cannot override {head!r} on {type(obj).__name__}")
    hints = get_type_hints(type(obj))
    if head not in hints:
        raise KeyError(f"{type(obj).__name__} has no field {head!r}")
    if rest:
        return dataclasses.replace(obj, **{head: _assign(getattr(obj, head), rest, raw)})
    return dataclasses.replace(obj, **{head: coerce(parse_literal(raw), hints[head])})


def _parse_like(existing: Any, raw: str) -> Any:
    value = parse_literal(raw)
    if isinstance(existing, tuple) and isinstance(value, list):
        return tuple(value)
    return value


def coerce(value: Any, hint: Any) -> Any:
    if hint is Any:
        return value
    origin = get_origin(hint)
    if origin in (Union, types.UnionType):
        for option in get_args(hint):
            try:
                return coerce(value, option)
            except (TypeError, ValueError):
                continue
        raise TypeError(f"Cannot coerce {value!r} to {hint}")
    if hint is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if hint is int and isinstance(value, int) and not isinstance(value, bool):
        return value
    if hint is bool and isinstance(value, bool):
        return value
    if hint is str and isinstance(value, str):
        return value
    if hint is slice and isinstance(value, slice):
        return value
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"Expected a sequence for {hint}, got {value!r}")
        args = get_args(hint)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(coerce(item, args[0]) for item in value)
        if args and len(args) == len(value):
            return tuple(coerce(item, arg) for item, arg in zip(value, args, strict=True))
        if not args:
            return tuple(value)
        raise TypeError(f"Expected {len(args)} items for {hint}, got {len(value)}")
    if origin is list:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"Expected a list for {hint}, got {value!r}")
        (item_type,) = get_args(hint) or (Any,)
        return [coerce(item, item_type) for item in value]
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError(f"Expected a dict for {hint}, got {value!r}")
        key_type, value_type = get_args(hint) or (Any, Any)
        return {coerce(key, key_type): coerce(item, value_type) for key, item in value.items()}
    if isinstance(value, hint):
        return value
    raise TypeError(f"Cannot coerce {value!r} to {hint}")


def parse_literal(text: str) -> Any:
    text = text.strip()
    if text.startswith("slice(") and text.endswith(")"):
        args = [_slice_bound(part) for part in _split_args(text[len("slice(") : -1])]
        return slice(*args)
    if text.startswith("[") and text.endswith("]"):
        return [parse_literal(part) for part in _split_args(text[1:-1])]
    if text.startswith("(") and text.endswith(")"):
        return tuple(parse_literal(part) for part in _split_args(text[1:-1]))
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("none", "null"):
        return None
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    if text.lstrip("+-").isdigit():
        return int(text)
    try:
        return float(text)
    except ValueError:
        return text


def _slice_bound(text: str) -> int | None:
    value = parse_literal(text)
    if value is None or isinstance(value, int):
        return value
    raise ValueError(f"Slice bounds must be integers, got {text!r}")


def _split_assignment(token: str) -> tuple[str, str]:
    if "=" not in token:
        raise ValueError(f"Expected key=value, got {token!r}")
    key, raw = token.split("=", 1)
    if not key:
        raise ValueError(f"Expected key=value, got {token!r}")
    return key, raw


def _split_args(text: str) -> list[str]:
    if not text.strip():
        return []
    parts: list[str] = []
    start = 0
    depth = 0
    quote = ""
    for index, char in enumerate(text):
        if quote:
            if char == quote:
                quote = ""
            continue
        if char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return parts


def _qualname(value: Any) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _import(path: str) -> Any:
    parts = path.split(".")
    for index in range(len(parts) - 1, 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:index]))
        except ModuleNotFoundError:
            continue
        obj = module
        for name in parts[index:]:
            obj = getattr(obj, name)
        return obj
    raise ImportError(f"Cannot import {path}")

