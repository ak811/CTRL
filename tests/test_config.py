from __future__ import annotations

from dataclasses import dataclass

import pytest

from common.config import build_config, config_to_dict, parse_config


@dataclass
class _Cfg:
    lr: float = 0.1
    steps: int = 10
    name: str = "x"
    flag: bool = False
    sizes: tuple[int, ...] = (4, 4)
    pairs: tuple[tuple[float, float], ...] = ((1.0, 2.0),)
    maybe: str | None = None


def test_precedence_defaults_yaml_cli(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("lr: 0.5\nsteps: 20\nsizes: [8, 8, 8]\npairs: [[3, 4]]\n")
    cfg, _ = parse_config(_Cfg, "t", ["--config", str(cfg_file), "--steps", "30", "--flag"])
    assert cfg.lr == 0.5  # from YAML
    assert cfg.steps == 30  # CLI beats YAML
    assert cfg.flag is True
    assert cfg.sizes == (8, 8, 8) and isinstance(cfg.sizes, tuple)
    assert cfg.pairs == ((3.0, 4.0),)
    assert cfg.name == "x"  # default


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        build_config(_Cfg, overrides={"nope": 1})


def test_roundtrip_to_dict():
    d = config_to_dict(_Cfg())
    assert d["sizes"] == [4, 4] and d["pairs"] == [[1.0, 2.0]]
    assert build_config(_Cfg, overrides=d) == _Cfg()


def test_project_configs_parse():
    from pathlib import Path

    from continual.config import ContinualConfig
    from meta.config import MetaConfig
    from transfer.config import TransferConfig

    root = Path(__file__).resolve().parents[1] / "configs"
    for cls, sub in ((ContinualConfig, "continual"), (MetaConfig, "meta"), (TransferConfig, "transfer")):
        for path in sorted((root / sub).glob("*.yaml")):
            build_config(cls, path)
