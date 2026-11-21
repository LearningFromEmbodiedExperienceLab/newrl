"""Match query patterns against scene labels."""

import re
from collections.abc import Sequence
from typing import Any


def resolve_matching_names(
    keys: str | Sequence[str],
    list_of_strings: Sequence[str],
    preserve_order: bool = False,
) -> tuple[list[int], list[str]]:
    """Match regular expressions against strings.

    Each target may match only one pattern, and every pattern must match at least
    once. Results follow the target order. ``preserve_order`` returns them in
    pattern order instead.
    """
    if isinstance(keys, str):
        keys = [keys]
    index_list: list[int] = []
    names_list: list[str] = []
    key_idx_list: list[int] = []
    matched_by = [None for _ in list_of_strings]
    keys_matched: list[list[str]] = [[] for _ in keys]
    for target_index, candidate in enumerate(list_of_strings):
        for key_index, pattern in enumerate(keys):
            if not re.fullmatch(pattern, candidate):
                continue
            if matched_by[target_index]:
                raise ValueError(
                    f"Multiple matches for {candidate!r}: "
                    f"{matched_by[target_index]!r} and {pattern!r}"
                )
            matched_by[target_index] = pattern
            index_list.append(target_index)
            names_list.append(candidate)
            key_idx_list.append(key_index)
            keys_matched[key_index].append(candidate)
    if preserve_order:
        order = sorted(range(len(index_list)), key=key_idx_list.__getitem__)
        index_list = [index_list[i] for i in order]
        names_list = [names_list[i] for i in order]
    if not all(keys_matched):
        missed = [pattern for pattern, found in zip(keys, keys_matched) if not found]
        raise ValueError(
            f"Patterns {missed} matched nothing. Available strings: {list(list_of_strings)}"
        )
    return index_list, names_list


def resolve_matching_names_values(
    data: dict[str, Any],
    list_of_strings: Sequence[str],
    preserve_order: bool = False,
    strict: bool = True,
) -> tuple[list[int], list[str], list[Any]]:
    """Match dictionary keys as patterns and return indices, names, and values."""
    if not isinstance(data, dict):
        raise TypeError(f"data must be a dictionary, got {type(data).__name__}")
    index_list: list[int] = []
    names_list: list[str] = [] 
    values_list: list[Any] = []
    key_idx_list: list[int] = []
    matched_by = [None for _ in list_of_strings]
    keys_matched: list[list[str]] = [[] for _ in data]
    items = list(data.items())
    for target_index, candidate in enumerate(list_of_strings):
        for key_index, (pattern, value) in enumerate(items):
            if not re.fullmatch(pattern, candidate):
                continue
            if matched_by[target_index]:
                raise ValueError(
                    f"Multiple matches for {candidate!r}: "
                    f"{matched_by[target_index]!r} and {pattern!r}"
                )
            matched_by[target_index] = pattern
            index_list.append(target_index)
            names_list.append(candidate)
            values_list.append(value)
            key_idx_list.append(key_index)
            keys_matched[key_index].append(candidate)
    if preserve_order:
        order = sorted(range(len(index_list)), key=key_idx_list.__getitem__)
        index_list = [index_list[i] for i in order]
        names_list = [names_list[i] for i in order]
        values_list = [values_list[i] for i in order]
    if strict and not all(keys_matched):
        missed = [pattern for pattern, found in zip(data, keys_matched) if not found]
        raise ValueError(
            f"Patterns {missed} matched nothing. Available strings: {list(list_of_strings)}"
        )
    return index_list, names_list, values_list


def resolve_scene_names(patterns: str | Sequence[str], labels: Sequence[str]) -> list[int]:
    """Match patterns against a scene label or its final path component.

    A pattern containing ``/`` is matched against the full label. Any other
    pattern is matched against the final component, so ``base`` selects
    ``robot/base``. Results are unique and follow scene order.
    """
    queries = (patterns,) if isinstance(patterns, str) else tuple(patterns)
    if not queries:
        raise ValueError("Selection must contain at least one pattern")
    leaves = [label.rsplit("/", 1)[-1] for label in labels]
    leaf_patterns = [pattern for pattern in queries if "/" not in pattern]
    full_patterns = [pattern for pattern in queries if "/" in pattern]
    selected: set[int] = set()
    if leaf_patterns:
        indices, _ = resolve_matching_names(leaf_patterns, leaves)
        selected.update(indices)
    if full_patterns:
        indices, _ = resolve_matching_names(full_patterns, labels)
        selected.update(indices)
    return sorted(selected)
