"""Typed experiment configuration.

Every experiment is described by a ``dataclass``. Values are resolved with the precedence

    dataclass defaults  <  YAML file (``--config``)  <  explicit CLI flags

CLI flags are generated automatically from the dataclass fields, so adding a field to a
config makes it immediately overridable from the command line (``foo_bar`` -> ``--foo-bar``).
Fields with nested/compound types (e.g. lists of tuples) are YAML-only.
"""

from __future__ import annotations

import argparse
import dataclasses
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

_SCALARS = (int, float, str)


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return tp, False


def _field_default(f: dataclasses.Field) -> Any:
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return f.default_factory()  # type: ignore[misc]
    return None


def _coerce(value: Any, tp: Any) -> Any:
    """Coerce YAML values (lists) into the declared field type (tuples), recursively."""
    tp, _ = _unwrap_optional(tp)
    if value is None:
        return None
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(v, args[0]) for v in value)
        if args:
            return tuple(_coerce(v, a) for v, a in zip(value, args))
        return tuple(value)
    if origin is list:
        return [_coerce(v, args[0]) if args else v for v in value]
    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def add_dataclass_arguments(parser: argparse.ArgumentParser, cls: type) -> None:
    """Register one CLI flag per scalar / flat-sequence dataclass field."""
    hints = typing.get_type_hints(cls)
    for f in dataclasses.fields(cls):
        tp, _ = _unwrap_optional(hints[f.name])
        flag = "--" + f.name.replace("_", "-")
        default = _field_default(f)
        help_text = f"(default: {default!r})"
        if tp is bool:
            parser.add_argument(
                flag, dest=f.name, action=argparse.BooleanOptionalAction, default=None, help=help_text
            )
        elif tp in _SCALARS:
            parser.add_argument(flag, dest=f.name, type=tp, default=None, help=help_text)
        elif typing.get_origin(tp) in (tuple, list):
            args = typing.get_args(tp)
            elem = args[0] if args else str
            if elem in _SCALARS:
                parser.add_argument(flag, dest=f.name, type=elem, nargs="+", default=None, help=help_text)
        # compound types are configurable through YAML only


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Config file {path} must contain a mapping at top level.")
    return dict(data)


def build_config(
    cls: type[T],
    yaml_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> T:
    """Instantiate ``cls`` from defaults, an optional YAML file and optional overrides."""
    values: dict[str, Any] = {}
    if yaml_path is not None:
        values.update(load_yaml(yaml_path))
    if overrides:
        values.update({k: v for k, v in overrides.items() if v is not None})

    names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(values) - names)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} option(s): {', '.join(unknown)}")

    hints = typing.get_type_hints(cls)
    coerced = {k: _coerce(v, hints[k]) for k, v in values.items()}
    return cls(**coerced)


def parse_config(
    cls: type[T],
    description: str,
    argv: Sequence[str] | None = None,
    extra_arguments: Callable[[argparse.ArgumentParser], None] | None = None,
) -> tuple[T, argparse.Namespace]:
    """Parse ``--config`` + auto-generated field flags (+ optional extra flags)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=None, help="YAML config file.")
    add_dataclass_arguments(parser, cls)
    if extra_arguments is not None:
        extra_arguments(parser)
    args = parser.parse_args(argv)
    overrides = {f.name: getattr(args, f.name, None) for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    return build_config(cls, args.config, overrides), args


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """Serialise a config dataclass into plain YAML/JSON-friendly Python types."""

    def _plain(v: Any) -> Any:
        if isinstance(v, (tuple, list)):
            return [_plain(x) for x in v]
        if isinstance(v, dict):
            return {k: _plain(x) for k, x in v.items()}
        return v

    return {k: _plain(v) for k, v in dataclasses.asdict(cfg).items()}
