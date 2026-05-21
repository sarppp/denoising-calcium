"""CIDC25 utilities: IO, statistics, noise model, losses, and models."""
from .config import Config, load_config
from .io import load_stack, stack_info, iter_frames
from .losses import anscombe_mse, calcium_kinetics_loss, poisson_gaussian_nll
from .noise import (
    FILE_NOISE,
    NOISE_LEVELS,
    NoiseParams,
    anscombe,
    identify_noise_level,
    inverse_anscombe,
    sample_poisson_gaussian,
)
from .stats import (
    mean_var_per_pixel,
    temporal_autocorr,
    estimate_poisson_gaussian,
)
from . import models
from .eval import StSNRResult, evaluate, denoise_stack, snr_spatial, snr_temporal, stsnr
from .models import build_model


def __getattr__(name: str):
    if name == "data":
        from . import data
        return data
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Config",
    "load_config",
    "load_stack",
    "stack_info",
    "iter_frames",
    "mean_var_per_pixel",
    "temporal_autocorr",
    "estimate_poisson_gaussian",
    "NoiseParams",
    "NOISE_LEVELS",
    "FILE_NOISE",
    "anscombe",
    "inverse_anscombe",
    "sample_poisson_gaussian",
    "identify_noise_level",
    "anscombe_mse",
    "poisson_gaussian_nll",
    "calcium_kinetics_loss",
    "models",
    "build_model",
    "data",
    "StSNRResult",
    "evaluate",
    "denoise_stack",
    "snr_spatial",
    "snr_temporal",
    "stsnr",
]
