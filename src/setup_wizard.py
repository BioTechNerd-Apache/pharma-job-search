"""AI-powered setup wizard: generates all configuration from a resume file."""

import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .config import PROJECT_ROOT, DEFAULT_CONFIG_PATH

# Service registration URLs
SERVICE_URLS = {
    "anthropic": "https://console.anthropic.com/settings/keys",
    "usajobs": "https://developer.usajobs.gov/APIRequest/Index",
    "adzuna": "https://developer.adzuna.com/",
    "jooble": "https://jooble.org/api/about",
}


@dataclass
class WizardOutput:
    """Holds all generated configuration from the wizard."""
    resume_profile: dict = field(default_factory=dict)
    search_terms: list[str] = field(default_factory=list)
    priority_terms: list[str] = field(default_factory=list)
    synonyms: dict[str, list[str]] = field(default_factory=dict)
    filter_include: list[str] = field(default_factory=list)
    filter_exclude: list[str] = field(default_factory=list)
    evaluator_patterns: dict[str, list[str]] = field(default_factory=dict)
    evaluator_prompt: dict[str, list[str]] = field(default_factory=dict)
    api_keys: dict[str, str] = field(default_factory=dict)


def _backup_file(path: Path):
    """Create a .bak backup of a file if it exists."""
    if path.is_file():
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, bak)
        print(f"  Backed up {path.name} -> {bak.name}")


def _prompt_yes_no(prompt: str, default: bool = True) -> bool:
    """Prompt user for yes/no, return bool."""
    suffix = " [Y/n] " if default else " [y/N] "
    try:
        answer = input(prompt + suffix).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    if not answer:
        return default
    return answer in ("y", "yes")


def _prompt_choice(prompt: str, choices: list[str]) -> str:
    """Prompt user to choose from a list."""
    for i, choice in enumerate(choices, 1):
        print(f"  {i}. {choice}")
    while True:
        try:
            answer = input(f"{prompt} [1-{len(choices)}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return choices[0]
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1]
        print(f"  Please enter a number 1-{len(choices)}")


def _load_existing_config() -> dict:
    """Load existing config.yaml if present."""
    if DEFAULT_CONFIG_PATH.is_file():
        with open(DEFAULT_CONFIG_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def _get_ai_client(api_key: str = "", provider: str = "anthropic",
                    model: str = "", base_url: str = ""):
    """Create an AIClient for wizard calls."""
    from .ai_client import AIClient
    return AIClient(provider=provider, model=model, api_key=api_key, base_url=base_url)


# ---------------------------------------------------------------------------
# Claude API Calls
# ---------------------------------------------------------------------------

def _load_example_profile() -> str:
    """Load the example resume profile as a reference."""
    example_path = PROJECT_ROOT / "data" / "resume_profile.example.json"
    if example_path.is_file():
        with open(example_path, "r") as f:
            return f.read()
    return "{}"


def call_generate_profile(client, resume_text: str) -> dict:
    """AI Call 1: Generate resume_profile.json from resume text."""
    example = _load_example_profile()

    system = """You are an expert career analyst for pharma/biotech scientists.
Extract a structured career profile from the resume text.
Be SPECIFIC — use actual instrument model names (e.g., "QuantStudio 7" not just "PCR"),
actual technique names (e.g., "8-color flow cytometry panel" not "cell analysis"),
and actual regulatory frameworks (e.g., "GLP method validation per ICH M10").

Output ONLY valid JSON, no markdown fences or explanation."""

    user_content = f"""Extract a structured resume profile from this resume.

EXAMPLE STRUCTURE (use as a template for the JSON keys and format):
{example}

RESUME TEXT:
{resume_text}

Generate the complete profile JSON with these keys:
- name, years_experience, target_level
- career_anchors (each position with period, title, key_skills)
- core_technical_platforms (specific instruments/techniques)
- regulatory_framework
- strongest_fit_domains, moderate_fit_domains, skip_domains
- never_claim (skills the candidate does NOT have based on resume gaps)

Be specific and technical. Infer never_claim from what's absent in the resume."""

    response = client.create_message(system=system, user_content=user_content, max_tokens=4096)

    raw = response.text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        retry = client.create_message(
            system="",
            user_content=f"Fix this invalid JSON and return ONLY valid JSON:\n{raw}",
            max_tokens=4096,
        )
        raw2 = retry.text.strip()
        if raw2.startswith("```"):
            raw2 = re.sub(r"^```(?:json)?\s*", "", raw2)
            raw2 = re.sub(r"\s*```$", "", raw2)
        return json.loads(raw2)


def call_generate_search_config(client, profile: dict) -> dict:
    """AI Call 2: Generate search terms, synonyms, and filters."""
    system = """You are an expert pharma/biotech job search strategist.
Generate search configuration tailored to the candidate's profile.
Output ONLY valid JSON, no markdown fences or explanation."""

    user_content = f"""Given this candidate profile, generate job search configuration:

PROFILE:
{json.dumps(profile, indent=2)}

Generate JSON with these keys:

1. "search_terms" (20-40 strings): Job search queries to use on Indeed/LinkedIn.
   Mix of broad ("scientist pharma") and specific ("bioanalytical scientist").
   Include role levels matching target_level.

2. "priority_terms" (8-12 strings): Subset of search_terms that are highest-priority
   (strongest fit domains). These get more results per search.

3. "synonyms" (8-15 groups): Dict mapping a search term to its aliases/variants.
   Example: {{"bioanalytical": ["bioanalysis", "ligand binding assay"]}}
   These auto-expand during search.

4. "filter_include" (100-200 strings): Keywords that a job title MUST partially match
   to be kept after scraping. Include the candidate's field terms, techniques,
   job title fragments. Use lowercase fragments (e.g., "biolog", "bioanalyt").

5. "filter_exclude" (200-400 strings): Keywords that cause a job to be EXCLUDED.
   Non-biology disciplines, unrelated roles, and clearly mismatched fields.
   Use lowercase fragments."""

    response = client.create_message(system=system, user_content=user_content, max_tokens=8192)

    raw = response.text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        retry = client.create_message(
            system="",
            user_content=f"Fix this invalid JSON and return ONLY valid JSON:\n{raw}",
            max_tokens=8192,
        )
        raw2 = retry.text.strip()
        if raw2.startswith("```"):
            raw2 = re.sub(r"^```(?:json)?\s*", "", raw2)
            raw2 = re.sub(r"\s*```$", "", raw2)
        return json.loads(raw2)


def call_generate_evaluator_patterns(client, profile: dict) -> dict:
    """AI Call 3: Generate evaluator regex patterns."""
    from .evaluator import (
        _BUILTIN_SKIP_TITLE_PATTERNS,
        _BUILTIN_SKIP_DESCRIPTION_PATTERNS,
        _BUILTIN_RESCUE_PATTERNS,
        _BUILTIN_BOOST_PATTERNS,
    )

    builtin_ref = {
        "skip_title_patterns_example": _BUILTIN_SKIP_TITLE_PATTERNS[:10],
        "rescue_patterns_example": _BUILTIN_RESCUE_PATTERNS[:5],
        "boost_patterns_example": _BUILTIN_BOOST_PATTERNS[:5],
    }

    system = """You are an expert at writing Python regex patterns for job evaluation.
All patterns use \\b word boundaries and are case-insensitive.
Output ONLY valid JSON, no markdown fences or explanation."""

    user_content = f"""Generate evaluator regex patterns tailored to this candidate profile:

PROFILE:
{json.dumps(profile, indent=2)}

REFERENCE PATTERNS (pharma example — use same style/format):
{json.dumps(builtin_ref, indent=2)}

Generate JSON with these keys:

1. "skip_title_patterns" (40-80 regex strings): Patterns that auto-skip job titles.
   Include: leadership roles too senior, wrong disciplines, entry-level roles,
   clinical/medical roles, sales, wrong technical specialties based on skip_domains.
   Use \\b word boundaries. Example: "\\bvp\\b", "\\bdata\\s+scientist\\b"

2. "skip_description_patterns" (10-20 regex strings): Patterns in descriptions
   that indicate the job is a poor fit (heavy emphasis on never_claim skills).
   Example: "\\bextensive\\b.*\\bexperience\\b.*\\bhplc\\b"

3. "rescue_patterns" (8-15 regex strings): If a job matches skip_description but
   ALSO matches a rescue pattern, do NOT skip it. These are the candidate's
   core skills. Example: "\\bqpcr\\b", "\\bflow\\s+cytometry\\b"

4. "boost_patterns" (10-20 regex strings): Patterns that flag jobs as high-priority
   for evaluation. Based on strongest_fit_domains.
   Example: "\\bbioanalytical\\b", "\\bgene\\s+therapy\\b"

IMPORTANT: All patterns must be valid Python regex. Use \\b for word boundaries."""

    response = client.create_message(system=system, user_content=user_content, max_tokens=8192)

    raw = response.text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        retry = client.create_message(
            system="",
            user_content=f"Fix this invalid JSON and return ONLY valid JSON:\n{raw}",
            max_tokens=8192,
        )
        raw2 = retry.text.strip()
        if raw2.startswith("```"):
            raw2 = re.sub(r"^```(?:json)?\s*", "", raw2)
            raw2 = re.sub(r"\s*```$", "", raw2)
        result = json.loads(raw2)

    # Validate all regex patterns compile
    for key in ("skip_title_patterns", "skip_description_patterns",
                "rescue_patterns", "boost_patterns"):
        patterns = result.get(key, [])
        valid = []
        for p in patterns:
            try:
                re.compile(p)
                valid.append(p)
            except re.error:
                pass  # Skip invalid patterns silently
        result[key] = valid

    return result


def call_generate_evaluator_prompt(client, profile: dict) -> dict:
    """AI Call 4: Generate personalized evaluator_prompt.yaml (domain calibration + scoring rules)."""

    # Load the existing file as a concrete example so the AI sees the exact format
    example_path = PROJECT_ROOT / "data" / "evaluator_prompt.yaml"
    example_yaml = ""
    if example_path.is_file():
        with open(example_path, "r") as f:
            example_yaml = f.read()

    system = """You are an expert at calibrating AI-powered job-fit scoring for pharma/biotech candidates.
Given a candidate profile, generate a personalized evaluator prompt config.
Output ONLY valid JSON with two keys: "domain_calibration" (list of strings) and "scoring_rules" (list of strings).
No markdown fences or explanation."""

    user_content = f"""Generate a personalized job-fit evaluator prompt config for this candidate.

CANDIDATE PROFILE:
{json.dumps(profile, indent=2)}

EXAMPLE OUTPUT FORMAT (for a different bioanalytical candidate — adapt to this candidate's actual background):
{example_yaml}

INSTRUCTIONS:

1. "domain_calibration" (12-20 strings): One line per domain category, with a score range.
   - Derive from the profile's strongest_fit_domains (65-80%), moderate_fit_domains (50-65%),
     and skip_domains (score "skip" or "usually skip").
   - Add nuanced sub-categories where the profile suggests partial overlap.
   - Format: "Domain description: score range" (e.g., "Flow cytometry / immunophenotyping: strong fit")

2. "scoring_rules" (5-9 strings): Scoring instructions for the AI evaluator.
   - ALWAYS include these two universal rules verbatim as the first two items:
     * "Score 0-100 based on overlap between candidate skills and STATED job requirements"
     * "fit_bucket: strong (70+), moderate (55-69), weak (40-54), poor (<40)"
     * "recommendation: apply (60+), maybe (45-59), skip (<45)"
     * "If the description is a generic company blurb with no specific job requirements, treat it as title-only"
     * "TITLE-ONLY or THIN DESCRIPTION: If the job has no description or only a brief/generic snippet with no specific requirements listed, cap the score at 50 maximum. Note \\"limited info — title-only assessment\\" in reasoning. Only match on what the title explicitly indicates. IMPORTANT: for domain_match, use the format \\"[Thin JD] <domain inferred from title only>\\" — never infer techniques or platforms from the candidate profile when the description is thin or generic."
   - Then add 2-4 PENALTY RULES specific to this candidate's gaps. Derive these from:
     * never_claim fields: if a candidate has never done X, any role where X is the PRIMARY skill should be capped at 35
     * skip_domains: roles in these domains should be capped at 35
     * Career-level mismatches (if applicable): roles clearly too junior/senior
   - Each penalty rule must: (a) name the role type clearly, (b) state what triggers it, (c) state the cap score, (d) explain why.
   - Keep penalty rules precise — avoid over-broad penalties that would wrongly cap borderline roles."""

    response = client.create_message(system=system, user_content=user_content, max_tokens=4096)

    raw = response.text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        retry = client.create_message(
            system="",
            user_content=f"Fix this invalid JSON and return ONLY valid JSON:\n{raw}",
            max_tokens=4096,
        )
        raw2 = retry.text.strip()
        if raw2.startswith("```"):
            raw2 = re.sub(r"^```(?:json)?\s*", "", raw2)
            raw2 = re.sub(r"\s*```$", "", raw2)
        result = json.loads(raw2)

    # Validate structure
    if not isinstance(result.get("domain_calibration"), list):
        result["domain_calibration"] = []
    if not isinstance(result.get("scoring_rules"), list):
        result["scoring_rules"] = []

    return result


# ---------------------------------------------------------------------------
# API Key Setup
# ---------------------------------------------------------------------------

def _check_api_key_status(config: dict) -> dict[str, str]:
    """Check which API keys are already configured."""
    status = {}

    # Anthropic
    eval_cfg = config.get("evaluation", {})
    key = eval_cfg.get("anthropic_api_key", "") or os.environ.get("ANTHROPIC_API_KEY", "")
    status["anthropic"] = key if key else ""

    # USAJobs
    usa_cfg = config.get("usajobs", {})
    key = usa_cfg.get("api_key", "") or os.environ.get("USAJOBS_API_KEY", "")
    email = usa_cfg.get("email", "") or os.environ.get("USAJOBS_EMAIL", "")
    status["usajobs_key"] = key if key else ""
    status["usajobs_email"] = email if email else ""

    # Adzuna
    adz_cfg = config.get("adzuna", {})
    app_id = adz_cfg.get("app_id", "") or os.environ.get("ADZUNA_APP_ID", "")
    app_key = adz_cfg.get("app_key", "") or os.environ.get("ADZUNA_APP_KEY", "")
    status["adzuna_app_id"] = app_id if app_id else ""
    status["adzuna_app_key"] = app_key if app_key else ""

    # Jooble
    joo_cfg = config.get("jooble", {})
    key = joo_cfg.get("api_key", "") or os.environ.get("JOOBLE_API_KEY", "")
    status["jooble"] = key if key else ""

    return status


def _prompt_api_keys(existing_config: dict) -> dict[str, str]:
    """Interactive API key setup. Returns dict of keys to save."""
    keys = {}
    status = _check_api_key_status(existing_config)

    print("\n--- API Key Setup ---")
    print("(Indeed and LinkedIn use JobSpy — no API key needed)\n")

    # Anthropic (required for evaluation + wizard)
    if status["anthropic"]:
        print(f"  Anthropic API key: configured (sk-...{status['anthropic'][-4:]})")
    else:
        print(f"  Anthropic API key: NOT configured")
        print(f"  Get one at: {SERVICE_URLS['anthropic']}")
        try:
            key = input("  Enter Anthropic API key (or press Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            key = ""
        if key:
            keys["anthropic_api_key"] = key

    # USAJobs
    if status["usajobs_key"]:
        print(f"\n  USAJobs API key: configured")
    else:
        print(f"\n  USAJobs API key: NOT configured (optional — adds federal jobs)")
        print(f"  Get one at: {SERVICE_URLS['usajobs']}")
        try:
            key = input("  Enter USAJobs API key (or press Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            key = ""
        if key:
            keys["usajobs_api_key"] = key
            if not status["usajobs_email"]:
                try:
                    email = input("  Enter USAJobs email: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    email = ""
                if email:
                    keys["usajobs_email"] = email

    # Adzuna
    if status["adzuna_app_id"] and status["adzuna_app_key"]:
        print(f"\n  Adzuna API: configured")
    else:
        print(f"\n  Adzuna API: NOT configured (optional — adds UK/US job aggregator)")
        print(f"  Get keys at: {SERVICE_URLS['adzuna']}")
        try:
            app_id = input("  Enter Adzuna App ID (or press Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            app_id = ""
        if app_id:
            keys["adzuna_app_id"] = app_id
            try:
                app_key = input("  Enter Adzuna App Key: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                app_key = ""
            if app_key:
                keys["adzuna_app_key"] = app_key

    # Jooble
    if status["jooble"]:
        print(f"\n  Jooble API key: configured")
    else:
        print(f"\n  Jooble API key: NOT configured (optional — adds Jooble aggregator)")
        print(f"  Get one at: {SERVICE_URLS['jooble']}")
        try:
            key = input("  Enter Jooble API key (or press Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            key = ""
        if key:
            keys["jooble_api_key"] = key

    return keys


# ---------------------------------------------------------------------------
# Save Wizard Output
# ---------------------------------------------------------------------------

def save_wizard_output(output: WizardOutput, existing_config: dict) -> list[str]:
    """Write all wizard-generated files. Returns list of files written."""
    files_written = []
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    # 1. Resume profile
    profile_path = data_dir / "resume_profile.json"
    _backup_file(profile_path)
    with open(profile_path, "w") as f:
        json.dump(output.resume_profile, f, indent=2)
    files_written.append(str(profile_path))

    # 2. Evaluator patterns YAML
    if output.evaluator_patterns:
        patterns_path = data_dir / "evaluator_patterns.yaml"
        _backup_file(patterns_path)
        with open(patterns_path, "w") as f:
            yaml.dump(output.evaluator_patterns, f, default_flow_style=False,
                      sort_keys=False, allow_unicode=True)
        files_written.append(str(patterns_path))

    # 2b. Evaluator prompt YAML (domain calibration + scoring rules)
    if output.evaluator_prompt:
        prompt_path = data_dir / "evaluator_prompt.yaml"
        _backup_file(prompt_path)
        with open(prompt_path, "w") as f:
            yaml.dump(output.evaluator_prompt, f, default_flow_style=False,
                      sort_keys=False, allow_unicode=True)
        files_written.append(str(prompt_path))

    # 3. Update config.yaml with search terms, filters, synonyms, and API keys
    config = dict(existing_config)

    # Ensure sections exist
    config.setdefault("search", {})
    config.setdefault("evaluation", {})
    config.setdefault("usajobs", {})
    config.setdefault("adzuna", {})
    config.setdefault("jooble", {})
    config.setdefault("output", {"directory": "data", "filename_prefix": "pharma_jobs", "formats": ["csv", "excel"]})
    config.setdefault("dashboard", {"port": 8501, "max_results": 2000})

    # Search config
    if output.search_terms:
        config["search"]["terms"] = output.search_terms
    if output.priority_terms:
        config["search"]["priority_terms"] = output.priority_terms
    if output.synonyms:
        config["search"]["synonyms"] = output.synonyms
    if output.filter_include:
        config["search"]["filter_include"] = output.filter_include
    if output.filter_exclude:
        config["search"]["filter_exclude"] = output.filter_exclude

    # Ensure search defaults
    config["search"].setdefault("sites", ["indeed", "linkedin"])
    config["search"].setdefault("days", 7)
    config["search"].setdefault("results_per_site", 100)
    config["search"].setdefault("priority_results_per_site", 200)
    config["search"].setdefault("delay_between_searches", 5)
    config["search"].setdefault("location", "United States")
    config["search"].setdefault("country_indeed", "USA")
    config["search"].setdefault("fetch_descriptions", False)

    # Evaluation config
    config["evaluation"].setdefault("provider", "anthropic")
    from .ai_client import DEFAULT_MODELS
    config["evaluation"].setdefault("model", DEFAULT_MODELS["anthropic"])
    config["evaluation"].setdefault("resume_profile", "data/resume_profile.json")
    config["evaluation"].setdefault("evaluations_store", "data/evaluations.json")
    config["evaluation"].setdefault("evaluator_patterns", "data/evaluator_patterns.yaml")
    config["evaluation"].setdefault("prompt_config", "data/evaluator_prompt.yaml")
    config["evaluation"].setdefault("max_concurrent", 5)
    config["evaluation"].setdefault("delay_between_calls", 0.5)
    config["evaluation"].setdefault("max_retries", 5)
    config["evaluation"].setdefault("description_max_chars", 6000)

    # API keys
    api_keys = output.api_keys
    if api_keys.get("anthropic_api_key"):
        config["evaluation"]["anthropic_api_key"] = api_keys["anthropic_api_key"]
    if api_keys.get("usajobs_api_key"):
        config["usajobs"]["api_key"] = api_keys["usajobs_api_key"]
    if api_keys.get("usajobs_email"):
        config["usajobs"]["email"] = api_keys["usajobs_email"]
    if api_keys.get("adzuna_app_id"):
        config["adzuna"]["app_id"] = api_keys["adzuna_app_id"]
    if api_keys.get("adzuna_app_key"):
        config["adzuna"]["app_key"] = api_keys["adzuna_app_key"]
    if api_keys.get("jooble_api_key"):
        config["jooble"]["api_key"] = api_keys["jooble_api_key"]

    _backup_file(DEFAULT_CONFIG_PATH)
    with open(DEFAULT_CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    files_written.append(str(DEFAULT_CONFIG_PATH))

    return files_written


# ---------------------------------------------------------------------------
# CLI Wizard Flow
# ---------------------------------------------------------------------------

def run_cli_wizard(resume_path: str) -> bool:
    """Run the full interactive CLI setup wizard.

    Returns True if setup completed successfully, False if cancelled.
    """
    print("=" * 60)
    print("  Pharma/Biotech Job Search — Setup Wizard")
    print("=" * 60)

    # Step 1: Validate inputs
    print("\nStep 1: Validating inputs...")
    resume_path = os.path.expanduser(resume_path)
    if not os.path.isfile(resume_path):
        print(f"  Error: File not found: {resume_path}")
        return False

    ext = os.path.splitext(resume_path)[1].lower()
    if ext not in (".pdf", ".docx", ".txt"):
        print(f"  Error: Unsupported file type '{ext}'. Use PDF, DOCX, or TXT.")
        return False
    print(f"  Resume file: {resume_path} ({ext})")

    # Load existing config
    existing_config = _load_existing_config()

    # Step 2: API key setup
    print("\nStep 2: API Key Setup")
    api_keys = _prompt_api_keys(existing_config)

    # Determine AI provider and key for wizard calls
    # Prefer dedicated wizard config; fall back to evaluation config
    wiz_cfg = existing_config.get("wizard", {})
    eval_cfg = existing_config.get("evaluation", {})

    if wiz_cfg.get("provider"):
        # Wizard has its own provider configured
        provider = wiz_cfg["provider"]
        model = wiz_cfg.get("model", "")
        base_url = wiz_cfg.get("base_url", "")
        wizard_api_key = (
            wiz_cfg.get("api_key", "")
            or os.environ.get({
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
            }.get(provider, ""), "")
        )
    else:
        # Fall back to evaluation config
        provider = eval_cfg.get("provider", "anthropic")
        model = eval_cfg.get("model", "")
        base_url = eval_cfg.get("base_url", "")
        wizard_api_key = (
            eval_cfg.get("api_key", "")
            or eval_cfg.get("anthropic_api_key", "")
            or os.environ.get({
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
            }.get(provider, ""), "")
        )

    # Newly entered Anthropic key from API key setup step overrides
    if api_keys.get("anthropic_api_key") and provider == "anthropic":
        wizard_api_key = api_keys["anthropic_api_key"]

    if not wizard_api_key and provider != "ollama":
        print(f"\n  Error: API key is required for the setup wizard (provider: {provider}).")
        if provider == "anthropic":
            print(f"  Get one at: {SERVICE_URLS['anthropic']}")
        try:
            wizard_api_key = input(f"  Enter {provider} API key: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Setup cancelled.")
            return False
        if not wizard_api_key:
            print("  Setup cancelled — API key is required.")
            return False
        api_keys["anthropic_api_key"] = wizard_api_key

    # Step 3: Extract resume text
    print("\nStep 3: Extracting resume text...")
    from .resume_parser import extract_text
    try:
        resume_text = extract_text(resume_path)
    except ValueError as e:
        print(f"  Error: {e}")
        return False
    print(f"  Extracted {len(resume_text):,} characters from resume")

    # Create AI client
    try:
        client = _get_ai_client(api_key=wizard_api_key, provider=provider,
                                model=model, base_url=base_url)
    except Exception as e:
        print(f"  Error creating AI client: {e}")
        return False

    output = WizardOutput(api_keys=api_keys)

    # Step 4: Generate resume profile
    print("\nStep 4: Generating resume profile...")
    try:
        profile = call_generate_profile(client, resume_text)
        output.resume_profile = profile
    except Exception as e:
        print(f"  Error generating profile: {e}")
        return False

    name = profile.get("name", "Unknown")
    years = profile.get("years_experience", "?")
    platforms = len(profile.get("core_technical_platforms", []))
    domains = len(profile.get("strongest_fit_domains", []))
    print(f"  Name: {name}")
    print(f"  Experience: {years} years")
    print(f"  Technical platforms: {platforms}")
    print(f"  Strongest fit domains: {domains}")

    if not _prompt_yes_no("\n  Accept this profile?"):
        choice = _prompt_choice("  What would you like to do?", ["Skip profile", "Cancel setup"])
        if choice == "Cancel setup":
            print("  Setup cancelled.")
            return False

    # Step 5: Generate search config
    print("\nStep 5: Generating search configuration...")
    try:
        search_config = call_generate_search_config(client, profile)
    except Exception as e:
        print(f"  Error generating search config: {e}")
        return False

    output.search_terms = search_config.get("search_terms", [])
    output.priority_terms = search_config.get("priority_terms", [])
    output.synonyms = search_config.get("synonyms", {})
    output.filter_include = search_config.get("filter_include", [])
    output.filter_exclude = search_config.get("filter_exclude", [])

    print(f"  Search terms: {len(output.search_terms)}")
    print(f"  Priority terms: {len(output.priority_terms)}")
    print(f"  Synonym groups: {len(output.synonyms)}")
    print(f"  Include filters: {len(output.filter_include)}")
    print(f"  Exclude filters: {len(output.filter_exclude)}")

    if not _prompt_yes_no("\n  Accept search configuration?"):
        choice = _prompt_choice("  What would you like to do?", ["Skip search config", "Cancel setup"])
        if choice == "Cancel setup":
            print("  Setup cancelled.")
            return False
        # Clear search config if skipped
        output.search_terms = []
        output.priority_terms = []
        output.synonyms = {}
        output.filter_include = []
        output.filter_exclude = []

    # Step 6: Generate evaluator patterns
    print("\nStep 6: Generating evaluator patterns...")
    try:
        patterns = call_generate_evaluator_patterns(client, profile)
        output.evaluator_patterns = patterns
    except Exception as e:
        print(f"  Error generating patterns: {e}")
        return False

    print(f"  Skip title patterns: {len(patterns.get('skip_title_patterns', []))}")
    print(f"  Skip description patterns: {len(patterns.get('skip_description_patterns', []))}")
    print(f"  Rescue patterns: {len(patterns.get('rescue_patterns', []))}")
    print(f"  Boost patterns: {len(patterns.get('boost_patterns', []))}")

    if not _prompt_yes_no("\n  Accept evaluator patterns?"):
        choice = _prompt_choice("  What would you like to do?", ["Skip patterns", "Cancel setup"])
        if choice == "Cancel setup":
            print("  Setup cancelled.")
            return False
        output.evaluator_patterns = {}

    # Step 7: Generate evaluator prompt (domain calibration + scoring rules)
    print("\nStep 7: Generating evaluator prompt config...")
    try:
        evaluator_prompt = call_generate_evaluator_prompt(client, profile)
        output.evaluator_prompt = evaluator_prompt
    except Exception as e:
        print(f"  Error generating evaluator prompt: {e}")
        return False

    print(f"  Domain calibration entries: {len(evaluator_prompt.get('domain_calibration', []))}")
    print(f"  Scoring rules: {len(evaluator_prompt.get('scoring_rules', []))}")

    if not _prompt_yes_no("\n  Accept evaluator prompt config?"):
        choice = _prompt_choice("  What would you like to do?", ["Skip prompt config", "Cancel setup"])
        if choice == "Cancel setup":
            print("  Setup cancelled.")
            return False
        output.evaluator_prompt = {}

    # Step 8: Save all files
    print("\nStep 8: Saving configuration files...")
    print("  Files to be written:")
    print(f"    - data/resume_profile.json")
    if output.evaluator_patterns:
        print(f"    - data/evaluator_patterns.yaml")
    if output.evaluator_prompt:
        print(f"    - data/evaluator_prompt.yaml")
    print(f"    - config.yaml (search terms, filters, API keys)")

    if not _prompt_yes_no("\n  Proceed with saving?"):
        print("  Setup cancelled — no files were written.")
        return False

    try:
        files = save_wizard_output(output, existing_config)
    except Exception as e:
        print(f"  Error saving files: {e}")
        return False

    print(f"\n  Saved {len(files)} files:")
    for f in files:
        print(f"    - {f}")

    # Reload evaluator patterns if we generated new ones
    if output.evaluator_patterns:
        from .evaluator import reload_patterns
        reload_patterns()

    print("\n" + "=" * 60)
    print("  Setup complete!")
    print(f"  Run 'pharma-job-search --days 1' to start searching.")
    print(f"  Run 'pharma-job-search --web' for the dashboard.")
    print("=" * 60)

    return True
