"""Resume profile loader — reads and validates the structured career profile JSON."""

import json
import logging
from pathlib import Path

from .config import EvaluationConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)

REQUIRED_KEYS = [
    "career_anchors",
    "core_technical_platforms",
    "regulatory_framework",
    "strongest_fit_domains",
    "never_claim",
]


def load_profile(config: EvaluationConfig) -> dict:
    """Load and validate the resume profile JSON.

    Returns:
        Parsed profile dict.

    Raises:
        FileNotFoundError: If the profile file doesn't exist.
        ValueError: If required keys are missing.
    """
    path = PROJECT_ROOT / config.resume_profile
    if not path.exists():
        raise FileNotFoundError(f"Resume profile not found: {path}")

    with open(path, "r") as f:
        profile = json.load(f)

    missing = [k for k in REQUIRED_KEYS if k not in profile]
    if missing:
        raise ValueError(f"Resume profile missing required keys: {missing}")

    logger.info(f"Loaded resume profile: {profile.get('name', 'Unknown')} "
                f"({len(profile.get('career_anchors', {}))} career anchors)")
    return profile


def profile_summary(config: EvaluationConfig) -> str:
    """Return a human-readable summary of the loaded profile."""
    profile = load_profile(config)
    lines = [
        f"Name: {profile.get('name', 'Unknown')}",
        f"Years Experience: {profile.get('years_experience', 'N/A')}",
        f"Target Levels: {', '.join(profile.get('target_level', []))}",
        f"Career Anchors: {', '.join(profile.get('career_anchors', {}).keys())}",
        f"Core Platforms: {len(profile.get('core_technical_platforms', []))}",
        f"Strong Fit Domains: {len(profile.get('strongest_fit_domains', []))}",
        f"Moderate Fit Domains: {len(profile.get('moderate_fit_domains', []))}",
        f"Never Claim Items: {len(profile.get('never_claim', []))}",
    ]
    return "\n".join(lines)
