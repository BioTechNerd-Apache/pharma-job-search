"""Configuration loader: YAML defaults + CLI argument overrides."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class SearchConfig:
    terms: List[str] = field(default_factory=lambda: ["bioanalytical scientist", "scientist"])
    sites: List[str] = field(
        default_factory=lambda: ["indeed", "linkedin", "zip_recruiter", "glassdoor", "google"]
    )
    days: int = 7
    results_per_site: int = 100
    delay_between_searches: int = 5
    location: str = "United States"
    country_indeed: str = "USA"
    fetch_descriptions: bool = False
    synonyms: Dict[str, List[str]] = field(default_factory=dict)
    filter_include: List[str] = field(default_factory=list)
    filter_exclude: List[str] = field(default_factory=list)


@dataclass
class USAJobsConfig:
    api_key: str = ""
    email: str = ""
    keywords: List[str] = field(
        default_factory=lambda: ["pharmaceutical", "biotech", "bioanalytical"]
    )

    @property
    def enabled(self) -> bool:
        key = self.api_key or os.environ.get("USAJOBS_API_KEY", "")
        email = self.email or os.environ.get("USAJOBS_EMAIL", "")
        return bool(key and email)

    def get_api_key(self) -> str:
        return self.api_key or os.environ.get("USAJOBS_API_KEY", "")

    def get_email(self) -> str:
        return self.email or os.environ.get("USAJOBS_EMAIL", "")


@dataclass
class AdzunaConfig:
    app_id: str = ""
    app_key: str = ""
    keywords: List[str] = field(
        default_factory=lambda: ["pharmaceutical", "biotech", "bioanalytical"]
    )
    results_per_page: int = 50
    max_pages: int = 5

    @property
    def enabled(self) -> bool:
        app_id = self.app_id or os.environ.get("ADZUNA_APP_ID", "")
        app_key = self.app_key or os.environ.get("ADZUNA_APP_KEY", "")
        return bool(app_id and app_key)

    def get_app_id(self) -> str:
        return self.app_id or os.environ.get("ADZUNA_APP_ID", "")

    def get_app_key(self) -> str:
        return self.app_key or os.environ.get("ADZUNA_APP_KEY", "")


@dataclass
class JoobleConfig:
    api_key: str = ""
    keywords: List[str] = field(
        default_factory=lambda: ["pharmaceutical", "biotech", "bioanalytical"]
    )
    max_pages: int = 5

    @property
    def enabled(self) -> bool:
        key = self.api_key or os.environ.get("JOOBLE_API_KEY", "")
        return bool(key)

    def get_api_key(self) -> str:
        return self.api_key or os.environ.get("JOOBLE_API_KEY", "")


@dataclass
class OutputConfig:
    directory: str = "data"
    filename_prefix: str = "pharma_jobs"
    formats: List[str] = field(default_factory=lambda: ["csv", "excel"])


@dataclass
class DashboardConfig:
    port: int = 8501
    max_results: int = 2000


@dataclass
class EvaluationConfig:
    anthropic_api_key: str = ""
    model: str = "claude-haiku-4-5-20251001"
    resume_profile: str = "data/resume_profile.json"
    evaluations_store: str = "data/evaluations.json"
    max_concurrent: int = 5
    delay_between_calls: float = 0.5
    max_retries: int = 5
    default_days: int = 1
    description_max_chars: int = 6000

    def get_api_key(self) -> str:
        return self.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    @property
    def enabled(self) -> bool:
        return bool(self.get_api_key())


@dataclass
class AppConfig:
    search: SearchConfig = field(default_factory=SearchConfig)
    usajobs: USAJobsConfig = field(default_factory=USAJobsConfig)
    adzuna: AdzunaConfig = field(default_factory=AdzunaConfig)
    jooble: JoobleConfig = field(default_factory=JoobleConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def load_yaml_config(path: Optional[Path] = None) -> dict:
    """Load config from YAML file, returning empty dict if not found."""
    config_path = path or DEFAULT_CONFIG_PATH
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def build_config(cli_args: Optional[dict] = None) -> AppConfig:
    """Build AppConfig from YAML defaults merged with CLI overrides."""
    raw = load_yaml_config()
    config = AppConfig()

    # Apply YAML values
    search_raw = raw.get("search", {})
    for key in ("terms", "sites", "days", "results_per_site", "delay_between_searches",
                "location", "country_indeed", "fetch_descriptions",
                "synonyms", "filter_include", "filter_exclude"):
        if key in search_raw:
            setattr(config.search, key, search_raw[key])

    usajobs_raw = raw.get("usajobs", {})
    for key in ("api_key", "email", "keywords"):
        if key in usajobs_raw:
            setattr(config.usajobs, key, usajobs_raw[key])

    adzuna_raw = raw.get("adzuna", {})
    for key in ("app_id", "app_key", "keywords", "results_per_page", "max_pages"):
        if key in adzuna_raw:
            setattr(config.adzuna, key, adzuna_raw[key])

    jooble_raw = raw.get("jooble", {})
    for key in ("api_key", "keywords", "max_pages"):
        if key in jooble_raw:
            setattr(config.jooble, key, jooble_raw[key])

    output_raw = raw.get("output", {})
    for key in ("directory", "filename_prefix", "formats"):
        if key in output_raw:
            setattr(config.output, key, output_raw[key])

    dashboard_raw = raw.get("dashboard", {})
    for key in ("port", "max_results"):
        if key in dashboard_raw:
            setattr(config.dashboard, key, dashboard_raw[key])

    evaluation_raw = raw.get("evaluation", {})
    for key in ("anthropic_api_key", "model", "resume_profile", "evaluations_store",
                "max_concurrent", "delay_between_calls", "max_retries", "default_days",
                "description_max_chars"):
        if key in evaluation_raw:
            setattr(config.evaluation, key, evaluation_raw[key])

    # Apply CLI overrides (non-None values only)
    if cli_args:
        if cli_args.get("terms"):
            config.search.terms = cli_args["terms"]
        if cli_args.get("days") is not None:
            config.search.days = cli_args["days"]
        if cli_args.get("location"):
            config.search.location = cli_args["location"]
        if cli_args.get("sites"):
            config.search.sites = cli_args["sites"]
        if cli_args.get("fetch_descriptions") is not None:
            config.search.fetch_descriptions = cli_args["fetch_descriptions"]
        if cli_args.get("extra_terms"):
            config.search.terms = config.search.terms + cli_args["extra_terms"]

    # Expand search terms using synonyms
    if config.search.synonyms:
        expanded = list(config.search.terms)
        for term in config.search.terms:
            term_lower = term.lower()
            for key, aliases in config.search.synonyms.items():
                if key.lower() in term_lower:
                    for alias in aliases:
                        if alias not in expanded:
                            expanded.append(alias)
        config.search.terms = expanded

    return config
