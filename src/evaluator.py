"""Job evaluation pipeline: Stage 1 rule-based pre-filter + Stage 2 Claude API scoring."""

import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from .config import EvaluationConfig, PROJECT_ROOT
from .exporter import get_master_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV loading / time-filtering helpers (used by CLI and dashboard)
# ---------------------------------------------------------------------------

def load_jobs_csv(config) -> pd.DataFrame:
    """Load the master jobs CSV."""
    master_path = get_master_path(config.output, "csv")
    if not master_path.exists():
        logger.error(f"Master CSV not found: {master_path}")
        logger.error("Run a search first: python job_search.py")
        return pd.DataFrame()
    df = pd.read_csv(master_path, parse_dates=["date_posted"])
    logger.info(f"Loaded {len(df)} jobs from {master_path}")
    return df


def filter_jobs_by_time(df: pd.DataFrame, eval_since: str = None,
                        eval_days: int = None, eval_all: bool = False,
                        default_days: int = 1) -> pd.DataFrame:
    """Filter jobs DataFrame by time window for evaluation."""
    if eval_all:
        logger.info(f"Evaluating all {len(df)} jobs")
        return df

    if "date_posted" not in df.columns:
        logger.warning("No date_posted column — evaluating all jobs")
        return df

    df["date_posted"] = pd.to_datetime(df["date_posted"], errors="coerce")

    if eval_since:
        cutoff = pd.Timestamp(eval_since)
    elif eval_days is not None:
        # Use start-of-day so "--eval-days 1" means "today and yesterday"
        cutoff = (pd.Timestamp.now().normalize() - pd.Timedelta(days=eval_days - 1))
    else:
        cutoff = (pd.Timestamp.now().normalize() - pd.Timedelta(days=default_days - 1))

    has_date = df[df["date_posted"] >= cutoff]
    no_date = df[df["date_posted"].isna()]
    filtered = pd.concat([has_date, no_date]).drop_duplicates()
    logger.info(f"Time filter: {len(has_date)} jobs since {cutoff.strftime('%Y-%m-%d %H:%M')} "
                f"+ {len(no_date)} with no date = {len(filtered)} total")
    return filtered


# ---------------------------------------------------------------------------
# Stage 1: Rule-based pre-filter
# ---------------------------------------------------------------------------

# Built-in fallback patterns (used when no evaluator_patterns.yaml exists)
_BUILTIN_SKIP_TITLE_PATTERNS = [
    # Senior leadership (too senior / wrong track)
    r"\bvp\b", r"\bsvp\b", r"\bceo\b", r"\bcfo\b", r"\bcoo\b", r"\bcmo\b",
    r"\bchief\b.*\bofficer\b",
    r"\bprogram\s+director\b",
    # Manufacturing / QC / QA
    r"\bqc analyst\b", r"\bqc technician\b", r"\bqc specialist\b",
    r"\bmanufacturing\b.*\btechnician\b", r"\bmanufacturing\b.*\boperator\b",
    r"\bproduction\b.*\boperator\b", r"\bproduction\b.*\btechnician\b",
    r"\bcgmp\b.*\bmanufactur\b", r"\bfill.*finish\b",
    r"\bdownstream\b", r"\bcompounding\b",
    # Instrumentation-specific mismatches
    r"\bhplc\b.*\bscientist\b", r"\blc-ms\b", r"\blcms\b", r"\bmass spec\b",
    r"\bprotein.*scientist\b", r"\bprotein.*chemist\b",
    r"\bformulation\b.*\bscientist\b",
    # Computational / bioinformatics
    r"\bbioinformatic", r"\bcomputational\b.*\bbiolog",
    r"\bdata\s+scientist\b", r"\bmachine\s+learning\b",
    r"\bsoftware\b.*\bengineer\b", r"\bai\b.*\bengineer\b",
    r"\bcomputer\s+scientist\b", r"\bcomputing\b", r"\bcyber\s+security\b",
    # Chemistry-specific
    r"\borganic\b.*\bchemist\b", r"\bprocess\b.*\bchemist\b",
    r"\bmedicinal\b.*\bchemist\b", r"\bsynthetic\b.*\bchemist\b",
    r"\banalytical\b.*\bchemist\b",
    # Entry level / trainee
    r"\bresearch\s+associate\b", r"\blab\s+technician\b",
    r"\bresearch\s+technician\b", r"\bintern\b", r"\bco-?op\b",
    r"\bpostdoc\b", r"\bpostdoctoral\b",
    r"\bassistant\s+prof",
    # Clinical / medical
    r"\bmedical\s+director\b", r"\bphysician\b", r"\bnurse\b",
    r"\bpharmacist\b", r"\bpharmacy\b",
    r"\bclinical\s+research\s+coordinator\b",
    r"\bclinical\s+research\s+associate\b", r"\bclinical\s+research\s+assistant\b",
    r"\bmedical\s+science\s+liaison\b",
    r"\boncology\s+expert\b", r"\boncology\s+data\b",
    r"\bmental\s+health\b", r"\bpsychologist\b", r"\bdietitian\b",
    # Sales / commercial
    r"\bsales\b", r"\bterritory\b.*\bmanager\b", r"\bbusiness\s+development\b",
    r"\bcommercial\b", r"\bbrand\s+marketing\b",
    # Other mismatches
    r"\bveterinar\b", r"\bregulatory\s+affairs\b.*\bspecialist\b",
    r"\bproject\s+manager\b", r"\bprogram\s+manager\b",
    r"\bsupply\s+chain\b", r"\bprocurement\b",
    r"\benvironmental\b.*\bscientist\b",
    # Unrelated fields
    r"\bfinance\b", r"\blaw\b", r"\bforestry\b", r"\benergy\b",
]

# Description patterns that auto-skip (unless rescued)
_BUILTIN_SKIP_DESCRIPTION_PATTERNS = [
    r"\bextensive\b.*\bexperience\b.*\bhplc\b",
    r"\bextensive\b.*\bexperience\b.*\blc-ms\b",
    r"\bbioreactor\s+operation\b",
    r"\bada\s+assay\b", r"\banti-drug\s+antibody\b",
    r"\bcell\s+line\s+development\b.*\bcho\b",
    r"\bcho\s+cell\b",
    r"\bin\s+vitro\s+transcription\b.*\bmrna\b",
    r"\bcleanroom\b.*\bexperience\b.*\brequired\b",
    r"\bformulation\b.*\bdevelopment\b.*\brequired\b",
    r"\bpharmacokinetic\s+modeling\b",
    r"\bpbpk\b.*\bmodel\b",
    r"\bprotein\s+purification\b.*\brequired\b",
    r"\bce-sds\b.*\brequired\b",
    r"\bsec-hplc\b.*\brequired\b",
    r"\bicief\b.*\brequired\b",
]

# Rescue patterns: if description matches these, do NOT skip even if skip_description matched
_BUILTIN_RESCUE_PATTERNS = [
    r"\bqpcr\b", r"\brt-qpcr\b", r"\bddpcr\b",
    r"\bflow\s+cytometry\b", r"\bfacs\b",
    r"\bgene\s+therapy\b", r"\bviral\s+vector\b",
    r"\baav\b", r"\blentivir\b", r"\bcar-t\b",
    r"\bglp\b", r"\bbiodistribution\b", r"\bshedding\b",
    r"\blnp\b", r"\bnucleic\s+acid\b",
    r"\borganoid\b", r"\b10x\s+genomics\b",
]

# Boost patterns: jobs matching these get priority for evaluation
_BUILTIN_BOOST_PATTERNS = [
    r"\bbioanalytical\b", r"\bbioanalysis\b",
    r"\bgene\s+therapy\b", r"\bcell\s+therapy\b",
    r"\bcar-t\b", r"\bcar\s+t\b",
    r"\bviral\s+vector\b", r"\baav\b",
    r"\bflow\s+cytometry\b", r"\bfacs\b",
    r"\bmethod\s+validation\b", r"\bassay\s+validation\b",
    r"\bqpcr\b", r"\bddpcr\b", r"\brt-qpcr\b",
    r"\bbiodistribution\b", r"\bshedding\b",
    r"\bglp\b", r"\btranslational\b.*\bbiomarker\b",
    r"\blnp\b", r"\bnucleic\s+acid\b.*\btherap\b",
    r"\borganoid\b", r"\bpotency\s+assay\b",
]


def _compile_patterns(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


# Public aliases for backward compatibility (tests and external code reference these)
SKIP_TITLE_PATTERNS = _BUILTIN_SKIP_TITLE_PATTERNS
SKIP_DESCRIPTION_PATTERNS = _BUILTIN_SKIP_DESCRIPTION_PATTERNS
RESCUE_PATTERNS = _BUILTIN_RESCUE_PATTERNS
BOOST_PATTERNS = _BUILTIN_BOOST_PATTERNS


def load_evaluator_patterns(yaml_path: str | None = None) -> dict[str, list[str]]:
    """Load evaluator patterns from YAML file, falling back to built-in defaults.

    Args:
        yaml_path: Path to evaluator_patterns.yaml. If None, uses default location.

    Returns:
        Dict with keys: skip_title_patterns, skip_description_patterns,
        rescue_patterns, boost_patterns — each a list of regex strings.
    """
    if yaml_path is None:
        yaml_path = str(PROJECT_ROOT / "data" / "evaluator_patterns.yaml")

    defaults = {
        "skip_title_patterns": list(_BUILTIN_SKIP_TITLE_PATTERNS),
        "skip_description_patterns": list(_BUILTIN_SKIP_DESCRIPTION_PATTERNS),
        "rescue_patterns": list(_BUILTIN_RESCUE_PATTERNS),
        "boost_patterns": list(_BUILTIN_BOOST_PATTERNS),
    }

    if not Path(yaml_path).is_file():
        return defaults

    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}
        result = {}
        for key in defaults:
            val = data.get(key)
            if isinstance(val, list) and val:
                # Validate each pattern compiles
                valid = []
                for p in val:
                    try:
                        re.compile(p)
                        valid.append(p)
                    except re.error:
                        logger.warning(f"Invalid regex in {key}: {p!r}, skipping")
                result[key] = valid
            else:
                result[key] = defaults[key]
        return result
    except Exception as e:
        logger.warning(f"Failed to load {yaml_path}: {e}, using built-in patterns")
        return defaults


def _init_compiled_patterns():
    """Load and compile all patterns. Called at import time and can be refreshed."""
    patterns = load_evaluator_patterns()
    return (
        _compile_patterns(patterns["skip_title_patterns"]),
        _compile_patterns(patterns["skip_description_patterns"]),
        _compile_patterns(patterns["rescue_patterns"]),
        _compile_patterns(patterns["boost_patterns"]),
    )


_skip_title_compiled, _skip_desc_compiled, _rescue_compiled, _boost_compiled = _init_compiled_patterns()


def reload_patterns(yaml_path: str | None = None):
    """Reload patterns from YAML (or built-in defaults). Call after wizard generates new patterns."""
    global _skip_title_compiled, _skip_desc_compiled, _rescue_compiled, _boost_compiled
    patterns = load_evaluator_patterns(yaml_path)
    _skip_title_compiled = _compile_patterns(patterns["skip_title_patterns"])
    _skip_desc_compiled = _compile_patterns(patterns["skip_description_patterns"])
    _rescue_compiled = _compile_patterns(patterns["rescue_patterns"])
    _boost_compiled = _compile_patterns(patterns["boost_patterns"])


@dataclass
class PreFilterResult:
    action: str  # "skip", "evaluate", "boost"
    reason: str
    matched_pattern: str = ""


def prefilter_job(title: str, description: str = "") -> PreFilterResult:
    """Stage 1: Rule-based pre-filter. Returns skip/evaluate/boost decision."""
    title = (title or "").strip()
    description = (description or "").strip()
    title_lower = title.lower()
    combined = f"{title} {description}"

    # Check title skip patterns
    for pattern in _skip_title_compiled:
        if pattern.search(title):
            return PreFilterResult("skip", "title_match", pattern.pattern)

    # Check description skip patterns (with rescue logic)
    if description:
        for pattern in _skip_desc_compiled:
            if pattern.search(description):
                # Check rescue patterns before skipping
                rescued = False
                for rescue in _rescue_compiled:
                    if rescue.search(combined):
                        rescued = True
                        break
                if not rescued:
                    return PreFilterResult("skip", "description_match", pattern.pattern)

    # Check boost patterns
    for pattern in _boost_compiled:
        if pattern.search(combined):
            return PreFilterResult("boost", "boost_match", pattern.pattern)

    return PreFilterResult("evaluate", "no_skip_pattern", "")


# ---------------------------------------------------------------------------
# Stage 2: Claude API evaluation
# ---------------------------------------------------------------------------

def load_resume_profile(config: EvaluationConfig) -> dict:
    """Load the structured resume profile JSON."""
    profile_path = PROJECT_ROOT / config.resume_profile
    with open(profile_path, "r") as f:
        return json.load(f)


def _load_prompt_config(prompt_config_path: str) -> dict:
    """Load domain calibration and scoring rules from YAML file."""
    path = Path(prompt_config_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / prompt_config_path
    if path.exists():
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _build_system_prompt(profile: dict, prompt_config_path: str = "data/evaluator_prompt.yaml") -> str:
    """Build the system prompt for Claude API evaluation."""
    prompt_cfg = _load_prompt_config(prompt_config_path)

    domain_lines = "\n".join(
        f"- {item}" for item in prompt_cfg.get("domain_calibration", [])
    )
    scoring_lines = "\n".join(
        f"- {item}" for item in prompt_cfg.get("scoring_rules", [])
    )

    return f"""You are a job-fit evaluator for a senior scientist in pharma/biotech.

CANDIDATE PROFILE:
{json.dumps(profile, indent=2)}

DOMAIN CALIBRATION (from historical fit assessments):
{domain_lines}

SCORING RULES:
{scoring_lines}

OUTPUT FORMAT — respond with ONLY valid JSON, no markdown:
{{
  "fit_score": <int 0-100>,
  "fit_bucket": "<strong|moderate|weak|poor>",
  "recommendation": "<apply|maybe|skip>",
  "domain_match": "<primary domain category>",
  "reasoning": "<2-3 sentence explanation>"
}}"""


def _is_substantive_description(description: str) -> bool:
    """Check if a description has real job requirements vs. generic company boilerplate.

    A substantive description should mention specific skills, qualifications,
    or responsibilities — not just "we are a global healthcare leader" type text.
    """
    if not description or len(description.strip()) < 200:
        return False
    desc_lower = description.lower()
    # Look for signals of real job content
    job_signals = [
        "qualifications", "requirements", "responsibilities", "experience",
        "must have", "preferred", "required", "skills", "duties",
        "bachelor", "master", "phd", "degree", "years of experience",
    ]
    return any(signal in desc_lower for signal in job_signals)


def _build_user_prompt(title: str, company: str, description: str,
                       description_available: bool) -> str:
    """Build the user prompt for a single job evaluation."""
    if description_available and description and _is_substantive_description(description):
        return f"""Evaluate this job for candidate fit:

TITLE: {title}
COMPANY: {company}

JOB DESCRIPTION:
{description}

Score based on overlap between candidate skills and STATED job requirements."""
    else:
        return f"""Evaluate this job for candidate fit (TITLE ONLY — no substantive description available, cap score at 50 max):

TITLE: {title}
COMPANY: {company}

NOTE: No specific job requirements are available. Score based ONLY on what the title explicitly indicates. Do NOT infer specific techniques unless the title mentions them."""


async def evaluate_single_job(
    client,
    system_prompt: str,
    title: str,
    company: str,
    description: str,
    description_max_chars: int,
    max_retries: int = 5,
) -> dict:
    """Evaluate a single job using the AI client with retry-with-backoff.

    Retries on rate limit (429) and overloaded errors with exponential backoff + jitter.
    The underlying SDK may have its own retries; this is an additional outer layer.
    """
    description_available = bool(description and str(description).strip()
                                  and str(description).lower() != "nan")
    truncated_desc = ""
    if description_available:
        truncated_desc = str(description).strip()[:description_max_chars]

    user_prompt = _build_user_prompt(title, company, truncated_desc, description_available)

    base_delay = 5.0  # Initial backoff delay in seconds

    for attempt in range(1, max_retries + 1):
        try:
            response = await asyncio.to_thread(
                client.create_message,
                system=system_prompt,
                user_content=user_prompt,
                max_tokens=1024,
            )

            raw_text = response.text.strip()
            # Strip markdown fences if present
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                raw_text = re.sub(r"\s*```$", "", raw_text)

            result = json.loads(raw_text)

            # Validate required fields
            required = ["fit_score", "fit_bucket", "recommendation", "domain_match", "reasoning"]
            for fld in required:
                if fld not in result:
                    result[fld] = "" if fld in ("domain_match", "reasoning") else 0

            # Ensure score is int
            result["fit_score"] = int(result.get("fit_score", 0))

            # Hard cap for title-only jobs — AI often ignores the prompt cap
            if not description_available:
                if result["fit_score"] > 50:
                    result["fit_score"] = 50
                # Recalculate bucket and recommendation from capped score
                s = result["fit_score"]
                result["fit_bucket"] = (
                    "strong" if s >= 70 else
                    "moderate" if s >= 55 else
                    "weak" if s >= 40 else
                    "poor"
                )
                result["recommendation"] = (
                    "apply" if s >= 60 else
                    "maybe" if s >= 45 else
                    "skip"
                )
                # Prefix domain_match so the user knows it's inferred
                dm = result.get("domain_match", "")
                if dm and not dm.startswith("[Title Only]"):
                    result["domain_match"] = f"[Title Only] {dm}"

            result["description_available"] = description_available
            result["input_tokens"] = response.input_tokens
            result["output_tokens"] = response.output_tokens
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error for '{title}': {e}")
            return _error_result(title, f"JSON parse error: {e}", description_available)

        except Exception as e:
            if client.is_rate_limit_error(e):
                if attempt == max_retries:
                    logger.warning(f"Rate limit exhausted after {max_retries} retries for '{title}'")
                    return _error_result(title, f"Rate limit after {max_retries} retries", description_available)
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 2)
                logger.info(f"Rate limited on '{title}' (attempt {attempt}/{max_retries}), "
                            f"backing off {delay:.1f}s")
                await asyncio.sleep(delay)
            elif client.is_overloaded_error(e):
                if attempt == max_retries:
                    logger.warning(f"API overloaded after {max_retries} retries for '{title}'")
                    return _error_result(title, f"API overloaded after {max_retries} retries", description_available)
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 2)
                logger.info(f"API overloaded for '{title}' (attempt {attempt}/{max_retries}), "
                            f"backing off {delay:.1f}s")
                await asyncio.sleep(delay)
            else:
                logger.warning(f"API error for '{title}': {e}")
                return _error_result(title, str(e), description_available)

    return _error_result(title, "Max retries exceeded", description_available)


def _error_result(title: str, error: str, description_available: bool) -> dict:
    return {
        "fit_score": 0,
        "fit_bucket": "error",
        "recommendation": "skip",
        "domain_match": "error",
        "reasoning": f"Evaluation failed: {error}",
        "description_available": description_available,
        "input_tokens": 0,
        "output_tokens": 0,
    }


async def evaluate_batch(
    jobs_df: pd.DataFrame,
    config: EvaluationConfig,
    progress_callback=None,
    store=None,
) -> list[dict]:
    """Evaluate a batch of jobs using the configured AI provider with rate limiting.

    Args:
        jobs_df: DataFrame with columns: job_url, title, company, description
        config: EvaluationConfig with API settings
        progress_callback: optional callable(completed, total) for progress updates
        store: optional EvaluationStore — when provided, each result is saved
               incrementally so interrupted runs don't lose progress

    Returns:
        List of dicts, each with job_url + evaluation fields
    """
    from .ai_client import create_ai_client

    if not config.enabled:
        provider = getattr(config, "provider", "anthropic")
        raise ValueError(f"No API key found for provider '{provider}'. "
                         f"Set the appropriate API key in config.yaml or environment variables.")

    client = create_ai_client(config)
    profile = load_resume_profile(config)
    system_prompt = _build_system_prompt(profile, config.prompt_config)

    results = []
    total = len(jobs_df)
    semaphore = asyncio.Semaphore(config.max_concurrent)
    completed = 0
    _unsaved_count = 0  # Track results not yet flushed to disk

    async def eval_with_rate_limit(row):
        nonlocal completed, _unsaved_count
        async with semaphore:
            result = await evaluate_single_job(
                client=client,
                system_prompt=system_prompt,
                title=str(row.get("title", "")),
                company=str(row.get("company", "")),
                description=str(row.get("description", "")),
                description_max_chars=config.description_max_chars,
                max_retries=config.max_retries,
            )
            result["job_url"] = row["job_url"]
            result["evaluated_timestamp"] = pd.Timestamp.now().isoformat()

            # Incremental persist: save to store immediately so refreshes
            # don't lose completed work
            if store is not None:
                job_url = result["job_url"]
                store.add_evaluation(job_url, {k: v for k, v in result.items()
                                               if k != "job_url"})
                _unsaved_count += 1
                # Flush to disk every 5 results to balance safety vs I/O
                if _unsaved_count >= 5:
                    store.save()
                    _unsaved_count = 0

            completed += 1
            if progress_callback:
                progress_callback(completed, total)
            await asyncio.sleep(config.delay_between_calls)
            return result

    tasks = [eval_with_rate_limit(row) for _, row in jobs_df.iterrows()]

    # Process in chunks to allow backpressure
    chunk_size = config.max_concurrent * 2
    for i in range(0, len(tasks), chunk_size):
        chunk = tasks[i:i + chunk_size]
        chunk_results = await asyncio.gather(*chunk, return_exceptions=True)
        for r in chunk_results:
            if isinstance(r, Exception):
                logger.error(f"Evaluation task failed: {r}")
            else:
                results.append(r)

    # Final flush for any remaining unsaved results
    if store is not None and _unsaved_count > 0:
        store.save()

    return results


def _save_descriptions_to_csv(fetched: dict, jobs_df: pd.DataFrame, config: EvaluationConfig):
    """Write fetched descriptions back to the master CSV so future runs skip re-fetching."""
    try:
        from .exporter import get_master_path
        from .config import PROJECT_ROOT
        # Build output config from evaluation config path
        # We need the output config — derive it from the CSV location
        csv_path = PROJECT_ROOT / "data" / "pharma_jobs.csv"
        if not csv_path.exists():
            return

        master_df = pd.read_csv(csv_path)

        # Map fetched descriptions by job_url from the jobs_df
        updated = 0
        for idx, desc in fetched.items():
            if idx in jobs_df.index:
                job_url = jobs_df.at[idx, "job_url"]
                mask = master_df["job_url"] == job_url
                if mask.any():
                    master_df.loc[mask, "description"] = desc
                    updated += 1

        if updated > 0:
            master_df.to_csv(csv_path, index=False)
            logger.info(f"Saved {updated} fetched descriptions back to master CSV")
    except Exception as e:
        logger.warning(f"Could not save descriptions to CSV: {e}")


def _update_eval_status_in_csv(evaluated_urls: list[str], skipped_urls: list[str]):
    """Write eval_status back to the master CSV for evaluated and skipped jobs."""
    try:
        csv_path = PROJECT_ROOT / "data" / "pharma_jobs.csv"
        if not csv_path.exists():
            return

        master_df = pd.read_csv(csv_path)
        if "eval_status" not in master_df.columns:
            master_df["eval_status"] = ""

        updated = 0
        if evaluated_urls:
            mask = master_df["job_url"].isin(evaluated_urls)
            master_df.loc[mask, "eval_status"] = "evaluated"
            updated += mask.sum()

        if skipped_urls:
            mask = master_df["job_url"].isin(skipped_urls)
            master_df.loc[mask, "eval_status"] = "skipped"
            updated += mask.sum()

        if updated > 0:
            master_df.to_csv(csv_path, index=False)
            logger.info(f"Updated eval_status in master CSV for {updated} jobs")
    except Exception as e:
        logger.warning(f"Could not update eval_status in CSV: {e}")


def run_evaluation_pipeline(
    jobs_df: pd.DataFrame,
    config: EvaluationConfig,
    store,
    prefilter_only: bool = False,
    re_evaluate: bool = False,
    progress_callback=None,
    desc_progress_callback=None,
) -> dict:
    """Run the full evaluation pipeline: pre-filter → skip evaluated → Claude API → persist.

    Args:
        jobs_df: Full DataFrame of jobs to evaluate
        config: EvaluationConfig
        store: EvaluationStore instance
        prefilter_only: If True, only run Stage 1 (no API calls)
        re_evaluate: If True, re-evaluate even if already scored
        progress_callback: optional callable(completed, total)
        desc_progress_callback: optional callable(fetched, total, succeeded) for description fetch

    Returns:
        Summary dict with counts
    """
    total = len(jobs_df)
    logger.info(f"Evaluation pipeline starting with {total} jobs")

    # Stage 0: Skip jobs already reviewed in the dashboard
    reviewed_skip_count = 0
    reviewed_skipped_urls = []
    try:
        reviewed_path = Path(__file__).resolve().parent.parent / "data" / "reviewed.json"
        if reviewed_path.exists():
            import json as _json
            with open(reviewed_path, "r") as f:
                reviewed_urls = set(_json.load(f).keys())
            remaining = []
            for _, row in jobs_df.iterrows():
                job_url = row.get("job_url", "")
                if job_url in reviewed_urls and not store.is_evaluated(job_url):
                    reviewed_skip_count += 1
                    reviewed_skipped_urls.append(job_url)
                    store.add_evaluation(job_url, {
                        "fit_score": 0,
                        "fit_bucket": "manually_reviewed",
                        "recommendation": "skip",
                        "domain_match": "N/A",
                        "reasoning": "Already reviewed manually \u2014 skipped retroactive evaluation",
                        "description_available": False,
                        "evaluated_timestamp": pd.Timestamp.now().isoformat(),
                        "input_tokens": 0,
                        "output_tokens": 0,
                    })
                else:
                    remaining.append(row)
            if reviewed_skip_count:
                jobs_df = pd.DataFrame(remaining)
                store.save()
                _update_eval_status_in_csv([], reviewed_skipped_urls)
                logger.info(f"Stage 0: {reviewed_skip_count} already-reviewed jobs stamped as manually_reviewed")
    except Exception as e:
        logger.warning(f"Could not check reviewed status: {e}")

    total = len(jobs_df)

    # Stage 1: Pre-filter
    skip_count = 0
    boost_count = 0
    evaluate_jobs = []
    skipped_urls = []  # Track URLs for eval_status writeback

    for _, row in jobs_df.iterrows():
        job_url = row.get("job_url", "")
        title = str(row.get("title", ""))
        description = str(row.get("description", ""))

        pf = prefilter_job(title, description)

        if pf.action == "skip":
            skip_count += 1
            skipped_urls.append(job_url)
            # Store the skip result
            store.add_evaluation(job_url, {
                "fit_score": 0,
                "fit_bucket": "prefilter_skip",
                "recommendation": "skip",
                "domain_match": "N/A",
                "reasoning": f"Pre-filter skip: {pf.reason} ({pf.matched_pattern})",
                "description_available": bool(description and description.lower() != "nan"),
                "evaluated_timestamp": pd.Timestamp.now().isoformat(),
                "input_tokens": 0,
                "output_tokens": 0,
            })
        else:
            evaluate_jobs.append((row, pf))
            if pf.action == "boost":
                boost_count += 1

    logger.info(f"Pre-filter: {skip_count} skipped, {boost_count} boosted, "
                f"{len(evaluate_jobs)} to evaluate")

    if prefilter_only:
        # Write eval_status for skipped jobs even in prefilter-only mode
        if skipped_urls:
            _update_eval_status_in_csv([], skipped_urls)
        return {
            "total": total,
            "prefilter_skipped": skip_count,
            "boosted": boost_count,
            "to_evaluate": len(evaluate_jobs),
            "evaluated": 0,
            "already_evaluated": 0,
            "reviewed_skipped": reviewed_skip_count,
        }

    # Filter out already-evaluated jobs (unless re_evaluate)
    already_evaluated = 0
    to_api = []
    for row, pf in evaluate_jobs:
        job_url = row.get("job_url", "")
        if not re_evaluate and store.is_evaluated(job_url):
            already_evaluated += 1
        else:
            to_api.append(row)

    # Sort: boosted jobs first
    boosted_urls = {row.get("job_url", "") for row, pf in evaluate_jobs if pf.action == "boost"}
    to_api_df = pd.DataFrame(to_api)
    if not to_api_df.empty and "job_url" in to_api_df.columns:
        to_api_df["_boost"] = to_api_df["job_url"].isin(boosted_urls)
        to_api_df = to_api_df.sort_values("_boost", ascending=False).drop(columns=["_boost"])

    logger.info(f"API evaluation: {len(to_api_df)} jobs ({already_evaluated} already evaluated, skipping)")

    if to_api_df.empty:
        # Write eval_status for skipped jobs
        if skipped_urls:
            _update_eval_status_in_csv([], skipped_urls)
        return {
            "total": total,
            "prefilter_skipped": skip_count,
            "boosted": boost_count,
            "to_evaluate": len(evaluate_jobs),
            "evaluated": 0,
            "already_evaluated": already_evaluated,
            "reviewed_skipped": reviewed_skip_count,
            "descriptions_fetched": 0,
        }

    # Stage 1.5: Fetch missing descriptions for jobs without substantive content
    descriptions_fetched = 0
    if not to_api_df.empty:
        needs_fetch = ~to_api_df["description"].apply(
            lambda d: _is_substantive_description(str(d) if pd.notna(d) else "")
        )
        fetch_count = needs_fetch.sum()
        if fetch_count > 0:
            from .description_fetcher import fetch_missing_descriptions

            def fetch_progress(done, total_fetch, succeeded):
                logger.info(f"  Fetching descriptions: {done}/{total_fetch} "
                            f"({succeeded} succeeded)")
                if desc_progress_callback:
                    desc_progress_callback(done, total_fetch, succeeded)

            fetched = fetch_missing_descriptions(
                to_api_df, needs_fetch,
                delay=1.5,
                progress_callback=fetch_progress,
            )
            # Update DataFrame with fetched descriptions
            for idx, desc in fetched.items():
                to_api_df.at[idx, "description"] = desc
            descriptions_fetched = len(fetched)
            logger.info(f"Enriched {descriptions_fetched}/{fetch_count} jobs with fetched descriptions")

            # Save fetched descriptions back to master CSV so future runs skip fetching
            if fetched:
                _save_descriptions_to_csv(fetched, jobs_df, config)

    # Stage 2: Claude API evaluation (store passed in for incremental saves)
    results = asyncio.run(evaluate_batch(to_api_df, config, progress_callback, store=store))

    # Results are already persisted incrementally — collect URLs for CSV writeback
    evaluated_urls = [r.get("job_url", "") for r in results if r.get("job_url")]

    # Write eval_status back to master CSV
    _update_eval_status_in_csv(evaluated_urls, skipped_urls)

    return {
        "total": total,
        "prefilter_skipped": skip_count,
        "boosted": boost_count,
        "to_evaluate": len(evaluate_jobs),
        "evaluated": len(results),
        "already_evaluated": already_evaluated,
        "reviewed_skipped": reviewed_skip_count,
        "descriptions_fetched": descriptions_fetched,
    }


def estimate_cost(jobs_df: pd.DataFrame, config: EvaluationConfig, store) -> dict:
    """Estimate the cost of evaluating a batch of jobs (dry run)."""
    total = len(jobs_df)
    skip_count = 0
    evaluate_count = 0
    already_evaluated = 0

    for _, row in jobs_df.iterrows():
        title = str(row.get("title", ""))
        description = str(row.get("description", ""))
        job_url = row.get("job_url", "")

        pf = prefilter_job(title, description)
        if pf.action == "skip":
            skip_count += 1
        elif store.is_evaluated(job_url):
            already_evaluated += 1
        else:
            evaluate_count += 1

    from .ai_client import get_model_pricing

    avg_input_tokens = 2500
    avg_output_tokens = 300

    provider = getattr(config, "provider", "anthropic")
    pricing = get_model_pricing(provider, config.model)
    if pricing:
        input_rate, output_rate = pricing
        input_cost = (evaluate_count * avg_input_tokens / 1_000_000) * input_rate
        output_cost = (evaluate_count * avg_output_tokens / 1_000_000) * output_rate
        total_cost = round(input_cost + output_cost, 4)
    else:
        total_cost = "N/A"

    return {
        "total_jobs": total,
        "prefilter_skip": skip_count,
        "already_evaluated": already_evaluated,
        "to_evaluate": evaluate_count,
        "estimated_input_tokens": evaluate_count * avg_input_tokens,
        "estimated_output_tokens": evaluate_count * avg_output_tokens,
        "estimated_cost_usd": total_cost,
    }
