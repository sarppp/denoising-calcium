"""Config loader for CIDC25 training and inference.

Design
------
One YAML schema for all 5 models (``deepinterp``, ``n2v3d``, ``deepcad``,
``mamba3d``, ``pinn``). The dispatch key is ``model.name``; model-specific
fields live under ``model.*`` and are ignored by other models.

Usage
-----
    from cidc.config import Config, load_config
    cfg = load_config("configs/n2v3d.yaml")
    model = cfg.build_model()                 # calls models.build_model(cfg)
    opt   = cfg.training.build_optimizer(model)

We deliberately use *dataclasses*, not pydantic/hydra:
- zero extra deps,
- trivial to serialise/deserialise,
- explicit about what fields exist,
- ``dataclasses.asdict`` for logging and checkpointing.

Unknown keys in the YAML are rejected (typo safety). Missing keys fall
back to the default in the dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

__all__ = [
    "Config",
    "load_config",
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "LossConfig",
    "InferenceConfig",
]


# --------------------------------------------------------------------------- #
# Individual sections                                                         #
# --------------------------------------------------------------------------- #


@dataclass
class ModelConfig:
    """Model architecture. ``name`` is the registry key.

    All architecture-specific fields go in ``kwargs`` so the config stays
    flat at the top level. Model builders pop what they need and warn on
    anything leftover.
    """

    name: str = "n2v3d"
    """One of: ``deepinterp``, ``n2v3d``, ``deepcad``, ``mamba3d``, ``pinn``."""

    kwargs: dict[str, Any] = field(default_factory=dict)
    """Model-specific constructor arguments."""


@dataclass
class GainAugConfig:
    enabled: bool = True
    log_uniform_range: tuple[float, float] = (20.0, 2000.0)
    prob: float = 0.5


@dataclass
class DataConfig:
    patch: tuple[int, int, int] = (32, 128, 128)     # (T, H, W)
    stride: tuple[int, int, int] = (8, 64, 64)
    batch_size: int = 16
    num_workers: int = 4
    samples_per_epoch: int = 10_000
    train_stacks: list[str] = field(default_factory=lambda: ["A1", "B1", "C2", "D2"])
    val_stacks: list[str] = field(default_factory=lambda: ["F1", "F2", "F3"])
    # Noisy stacks scored against ref_stack. F3 is OOD (Task 2).
    # Never include the ref_stack here — it is clean ground truth.
    ref_stack: str = "F0"
    # Clean reference stack. Lives in the val/ directory. Never trained on.
    gain_aug: GainAugConfig = field(default_factory=GainAugConfig)
    flip: bool = True
    rot90: bool = True
    temporal_reverse: bool = True


@dataclass
class EarlyStopConfig:
    metric: str = "stsnr_val"
    patience: int = 5
    higher_is_better: bool = True


@dataclass
class TrainingConfig:
    optimizer: str = "adamw"
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    scheduler: str = "cosine_restarts"
    restarts: int = 3
    epochs: int = 50
    warmup_steps: int = 500
    grad_clip: float = 1.0
    grad_accum: int = 1                        # gradient accumulation steps
    grad_ckpt: bool = False                    # activation checkpointing (memory save)
    ema_decay: float = 0.999
    amp: bool = True                           # mixed precision (bf16 where possible)
    seed: int = 0
    early_stop: EarlyStopConfig = field(default_factory=EarlyStopConfig)
    log_every: int = 50
    ckpt_every: int = 1                        # in epochs


@dataclass
class AuxLossConfig:
    """Single auxiliary loss entry (PINN, DeepInterp, etc.)."""

    enabled: bool = False
    weight: float = 0.0
    # Free-form extras (e.g. sparsity_l1 for PINN).
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossConfig:
    name: str = "poisson_gaussian_nll"  # poisson_gaussian_nll | anscombe_mse | mse | mae | huber
    var_floor: float = 1.0
    huber_delta: float = 1.0            # δ for Huber loss; ignored by other losses
    aux: dict[str, AuxLossConfig] = field(default_factory=dict)
    # ``aux`` is a dict keyed by loss name; empty = no aux losses.


@dataclass
class TTAConfig:
    rotations: int = 4        # 1, 2, or 4 (90° rotations of the HW plane)
    flips: bool = True        # horizontal + vertical → up to ×4 with rotations


@dataclass
class InferenceConfig:
    tile: tuple[int, int, int] = (32, 128, 128)
    overlap: tuple[int, int, int] = (8, 16, 16)
    tta: TTAConfig = field(default_factory=TTAConfig)
    # Output denormalisation is handled in-model (Anscombe inverse).


# --------------------------------------------------------------------------- #
# Top-level                                                                   #
# --------------------------------------------------------------------------- #


@dataclass
class Config:
    name: str = "unnamed"
    """Free-form run name used for checkpoints and logs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def build_model(self):
        """Instantiate the model via the registry."""
        from .models import build_model
        return build_model(self.model)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# YAML <-> dataclass                                                          #
# --------------------------------------------------------------------------- #


def _coerce(dc_cls, raw: dict[str, Any]):
    """Recursively construct a dataclass from a dict, rejecting unknown keys."""
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict for {dc_cls.__name__}, got {type(raw).__name__}")
    known: dict[str, Any] = {}
    legal = {f.name for f in fields(dc_cls)}
    unknown = set(raw) - legal
    if unknown:
        raise ValueError(
            f"Unknown keys for {dc_cls.__name__}: {sorted(unknown)}. "
            f"Allowed: {sorted(legal)}."
        )
    # Resolve forward-reference string annotations to real types.
    hints = get_type_hints(dc_cls)
    for f in fields(dc_cls):
        if f.name not in raw:
            continue
        v = raw[f.name]
        ftype = hints.get(f.name, f.type)
        type_str = str(ftype)
        # Nested dataclass?
        if is_dataclass(ftype):
            known[f.name] = _coerce(ftype, v)
        # dict[str, AuxLossConfig] special case.
        elif f.name == "aux" and isinstance(v, dict):
            known[f.name] = {k: _coerce(AuxLossConfig, sub) for k, sub in v.items()}
        # Tuple coercion (YAML gives list).
        elif isinstance(v, list) and "tuple" in type_str:
            known[f.name] = tuple(v)
        else:
            known[f.name] = v
    return dc_cls(**known)


def load_config(path: str | Path) -> Config:
    """Load a YAML config and coerce into :class:`Config`.

    Raises
    ------
    ValueError
        On any unknown key (typo safety).
    """
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}
    return _coerce(Config, raw)
