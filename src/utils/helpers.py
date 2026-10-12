"""
helpers.py – Configuration loader, logger setup, and path utilities.
"""

import os
import sys
import logging
import yaml
from pathlib import Path
from dotenv import load_dotenv


# ── Project root is two levels up from src/utils/ ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return PROJECT_ROOT


def load_config(config_path: str | None = None) -> dict:
    """
    Load the YAML configuration file.

    Parameters
    ----------
    config_path : str, optional
        Absolute or project-relative path.  Defaults to ``configs/config.yaml``.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path

    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def load_env() -> None:
    """Load environment variables from the project ``.env`` file."""
    env_path = PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=env_path)


def setup_logger(
    name: str = "cctv_assistant",
    config: dict | None = None,
) -> logging.Logger:
    """
    Configure and return a named logger.

    The logger writes to both *stdout* and the log file specified in the
    configuration (``logging.log_file``).

    Parameters
    ----------
    name : str
        Logger name.
    config : dict, optional
        Full configuration dict.  If ``None``, defaults are used.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger                           # already configured

    level_str = "INFO"
    log_file = PROJECT_ROOT / "data" / "logs" / "cctv_assistant.log"

    if config and "logging" in config:
        level_str = config["logging"].get("level", "INFO")
        log_file = PROJECT_ROOT / config["logging"].get(
            "log_file", "data/logs/cctv_assistant.log"
        )

    level = getattr(logging, level_str.upper(), logging.INFO)
    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_file))
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def resolve_path(relative: str) -> Path:
    """Resolve a project-relative path to an absolute ``Path``."""
    p = Path(relative)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (including parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
