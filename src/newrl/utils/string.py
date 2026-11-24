"""Resolve exact names or full-match regexes against scene labels."""

import re
from collections.abc import Sequence
from typing import Any


def _resolve_names(keys, names, preserve_order, *, strict=True, scene_labels=False):
    queries = (keys,) if isinstance(keys, str) else tuple(keys)
    patterns = []
    for query in queries:
        try:
            patterns.append(re.compile(query))
        except re.error as error:
            raise ValueError(f'Invalid name pattern {query!r}: {error}') from error
    matches = [[] for _ in queries]
    selected = []
    for index, name in enumerate(names):
        matched = None
        for query_index, pattern in enumerate(patterns):
            candidate = name.rsplit('/', 1)[-1] if scene_labels and '/' not in queries[query_index] else name
            if pattern.fullmatch(candidate) is None:
                continue
            if matched is not None:
                raise ValueError(f'Multiple patterns match {name!r}: {queries[matched]!r} and {queries[query_index]!r}')
            matched = query_index
            matches[query_index].append(name)
        if matched is not None:
            selected.append((index, matched))
    if strict and not all(matches):
        missed = [query for query, found in zip(queries, matches, strict=True) if not found]
        raise ValueError(f'Patterns {missed} matched nothing. Available names: {list(names)}. Patterns must match whole names.')
    if preserve_order:
        selected.sort(key=lambda pair: pair[1])
    return [index for index, _ in selected], [query for _, query in selected]


def resolve_matching_names(
    keys: str | Sequence[str],
    list_of_strings: Sequence[str],
    preserve_order: bool = False,
) -> tuple[list[int], list[str]]:
    """Resolve names/regexes in scene order, or query order when requested.

    Every query must match; a name matching multiple queries is an error.
    Regexes use fullmatch, so use ``.*`` to match a prefix or suffix.
    """
    indices, _ = _resolve_names(keys, list_of_strings, preserve_order)
    return indices, [list_of_strings[index] for index in indices]


def resolve_matching_names_values(
    data: dict[str, Any],
    list_of_strings: Sequence[str],
    preserve_order: bool = False,
    strict: bool = True,
) -> tuple[list[int], list[str], list[Any]]:
    """Resolve dictionary patterns with the same ordering and ambiguity rules."""
    if not isinstance(data, dict):
        raise TypeError(f'data must be a dictionary, got {type(data).__name__}')
    indices, queries = _resolve_names(tuple(data), list_of_strings, preserve_order, strict=strict)
    values = tuple(data.values())
    return indices, [list_of_strings[index] for index in indices], [values[query] for query in queries]


def resolve_scene_names(
    patterns: str | Sequence[str], labels: Sequence[str], preserve_order: bool = False,
) -> list[int]:
    """Resolve scene names or regexes, accepting unqualified leaf names.

    Queries containing ``/`` match full labels; others match the final path
    component (``base`` selects ``robot/base``). Preserve scene order by default,
    or query order with ``preserve_order=True``. Every query must match and
    overlapping queries are rejected, including overlaps across full/leaf names.
    """
    queries = (patterns,) if isinstance(patterns, str) else tuple(patterns)
    if not queries:
        raise ValueError('Selection must contain at least one pattern')
    return _resolve_names(queries, labels, preserve_order, scene_labels=True)[0]
