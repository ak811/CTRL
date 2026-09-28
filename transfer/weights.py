"""Parameter transfer between SB3 actor-critic policies with different action spaces.

Only modules whose shapes are task-agnostic are transferred:

* ``encoder``     - the CNN feature extractor,
* ``encoder_mlp`` - the feature extractor plus the policy/value MLP trunks.

Action heads (``action_net``, ``value_net``, ``log_std``) are always re-initialised, since
Snake (Discrete(4)), Pong (Discrete(3)) and PuckWorld (Box(2)) are incompatible there.
"""

from __future__ import annotations

from pathlib import Path

import torch
from stable_baselines3.common.policies import ActorCriticPolicy

PREFIXES: dict[str, tuple[str, ...]] = {
    "encoder": ("features_extractor.", "pi_features_extractor.", "vf_features_extractor."),
    "encoder_mlp": (
        "features_extractor.",
        "pi_features_extractor.",
        "vf_features_extractor.",
        "mlp_extractor.",
    ),
}


def load_policy_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load policy parameters from an SB3 ``.zip`` or a raw ``state_dict`` ``.pt`` file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".zip":
        from stable_baselines3.common.save_util import load_from_zip_file

        _, params, _ = load_from_zip_file(path, device="cpu")
        if params is None or "policy" not in params:
            raise ValueError(f"{path} does not contain policy parameters")
        return params["policy"]
    return torch.load(path, map_location="cpu", weights_only=True)


def transfer_weights(
    policy: ActorCriticPolicy, source_state: dict[str, torch.Tensor], layers: str = "encoder"
) -> list[str]:
    """Copy selected source parameters into ``policy``. Returns the transferred keys."""
    if layers not in PREFIXES:
        raise ValueError(f"layers must be one of {tuple(PREFIXES)}")
    target_state = policy.state_dict()
    selected: dict[str, torch.Tensor] = {}
    mismatched: list[str] = []
    for key, value in source_state.items():
        if not key.startswith(PREFIXES[layers]) or key not in target_state:
            continue
        if target_state[key].shape != value.shape:
            mismatched.append(
                f"{key}: source {tuple(value.shape)} vs target {tuple(target_state[key].shape)}"
            )
            continue
        selected[key] = value
    if mismatched:
        raise ValueError("Shape mismatch while transferring weights:\n  " + "\n  ".join(mismatched))
    if not selected:
        raise ValueError("No parameters matched; is the source model built with the same encoder?")
    policy.load_state_dict(selected, strict=False)
    return sorted(selected)


def freeze_encoder(policy: ActorCriticPolicy) -> int:
    """Disable gradients for the feature extractor(s). Returns the number of frozen tensors."""
    modules = {
        id(m): m
        for m in (
            policy.features_extractor,
            getattr(policy, "pi_features_extractor", None),
            getattr(policy, "vf_features_extractor", None),
        )
        if m is not None
    }
    frozen = 0
    for module in modules.values():
        for p in module.parameters():
            if p.requires_grad:
                p.requires_grad_(False)
                frozen += 1
    return frozen
