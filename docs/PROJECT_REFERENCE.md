# Pharma/Biotech Job Search Tool — Complete Project Reference

> **Purpose:** This file preserves the entire codebase so you can reconstruct the project in a new Claude Code session if the chat becomes too long for compacting.
>
> **Owner:** BioTechNerd — CGT, Bioanalytical, Molecular Biology, Oncology
>
> **Created:** 2026-02-10
> **Updated:** 2026-02-19

---

## Project Summary

A Python CLI + Streamlit dashboard tool that aggregates pharma/biotech job listings from **5 sources** (Indeed, LinkedIn via JobSpy; USAJobs, Adzuna, Jooble via their APIs), with an **AI-powered evaluation pipeline** that scores jobs against the candidate's resume profile. Features:

- **Parallel scraping** — all 5 scrapers run concurrently via `ThreadPoolExecutor`
- **Synonym expansion** — search terms auto-expand (e.g. "cell gene therapy" also searches "CAR-T scientist", "gene therapy", etc.)
- **Discipline filtering** — include/exclude keyword lists keep only biology/medicine roles, removing sales, IT, nursing, AI/ML, etc.
  - **Both** include and exclude filters match against job **title only** (avoids false positives from keywords in descriptions)
- **3-layer deduplication** — exact URL match, fuzzy (company+title+state), cross-source (same title + similar company across boards)
- **Cross-source company matching** — normalizes company names (strips Inc/LLC/Corp/etc.) and recognizes known job boards/recruiters as duplicates
- **Reviewed-job preservation** — reviewed jobs get +1000 richness boost so they always win dedup
- **Repost detection** — tracks when the same job appears with different dates
- **Master CSV/Excel** — single rolling file, new results merge with existing data
- **AI evaluation pipeline** — 2-stage design:
  - Stage 1: Rule-based pre-filter (skip/boost/evaluate) with rescue logic — no API cost
  - Stage 1.5: Fetches full job descriptions from URLs for jobs with missing/thin descriptions
  - Stage 2: Claude Haiku API scoring against resume profile (fit_score 0-100, recommendation, matching/missing skills); title-only jobs hard-capped at 50 with `[Title Only]` domain_match prefix
- **Evaluation persistence** — results stored in `evaluations.json` keyed by job_url to prevent re-evaluation; incremental saves every 5 jobs during evaluation
- **Streamlit dashboard** — AG Grid with 3 tabs: Job Listings + Evaluation Results (with "Info" column for title-only status + sidebar filter) + Setup, with review tracking, color-coded fit scores
- **iCloud sync** — `.command` launcher forces iCloud download before opening dashboard
- **CLI flags** — `--days`, `--terms`, `--extra-terms`, `--sites`, `--web`, `--reprocess`, `--evaluate`, `--evaluate-only`, `--eval-days`, `--eval-all`, `--eval-prefilter-only`, `--eval-dry-run`, `--eval-summary`, `--eval-export`, `--eval-min-score`, `--re-evaluate`

### Architecture

```
pharma-job-search/
├── job_search.py              # CLI entry point (search + evaluation + dashboard)
├── config.yaml                # Search terms, API keys, filters, synonyms, evaluation config
├── requirements.txt           # Python dependencies
├── BioTechNerd-Apache's Job Search.command  # Double-click launcher (dashboard only)
├── CLAUDE.md                  # Project instructions for Claude Code
├── PROJECT_REFERENCE.md       # This file
├── data/                      # Output directory
│   ├── pharma_jobs.csv        # Master CSV (rolling)
│   ├── pharma_jobs.xlsx       # Master Excel (rolling)
│   ├── pharma_jobs_raw.csv    # Raw pre-filter/dedup data (for --reprocess)
│   ├── reviewed.json          # Review status tracking
│   ├── evaluations.json       # AI evaluation results (keyed by job_url)
│   └── resume_profile.json    # Structured candidate profile for AI evaluation
├── src/
│   ├── __init__.py            # (empty)
│   ├── config.py              # YAML loader + dataclasses (AppConfig, EvaluationConfig, etc.)
│   ├── aggregator.py          # Orchestrator: ThreadPoolExecutor(5), normalize, filter
│   ├── dedup.py               # 3-layer dedup: URL, fuzzy, cross-source
│   ├── exporter.py            # CSV/Excel merge-on-save to single master file
│   ├── dashboard.py           # Streamlit UI — 2 tabs: Job Listings + Evaluations
│   ├── scraper_jobspy.py      # Indeed/LinkedIn via python-jobspy (per-site)
│   ├── scraper_usajobs.py     # USAJobs API client
│   ├── scraper_adzuna.py      # Adzuna API client
│   ├── scraper_jooble.py      # Jooble API client
│   ├── evaluator.py           # 2-stage evaluation: pre-filter + Claude API scoring
│   ├── eval_persistence.py    # EvaluationStore: JSON persistence for evaluations
│   ├── description_fetcher.py # Fetches job descriptions from URLs via HTML scraping
│   └── resume_profile.py      # Resume profile loader + validator
└── tests/
    ├── __init__.py            # (empty)
    └── test_evaluator.py      # Tests for Stage 1 pre-filter logic
```

### Dependencies (requirements.txt)

```
python-jobspy>=1.1.0
pandas>=2.0.0
openpyxl>=3.1.0
streamlit>=1.30.0
streamlit-aggrid>=1.0.0
pyyaml>=6.0
requests>=2.31.0
beautifulsoup4>=4.12.0
anthropic>=0.40.0
```

---

## File: job_search.py

```python
#!/usr/bin/env python3
"""Pharma/Biotech Job Search Aggregator - CLI entry point."""

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.config import build_config, PROJECT_ROOT
from src.aggregator import run_search, reprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pharma/Biotech Job Search Aggregator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python job_search.py                          # Run search, save to master CSV
  python job_search.py --days 1                 # Last 24hrs
  python job_search.py --reprocess              # Re-run filter/dedup from raw data (no scraping)
  python job_search.py --web                    # Open dashboard only (no new scrape)
  python job_search.py --terms "DMPK scientist" "bioanalytical" --days 7
  python job_search.py --extra-terms "viral vector" "CAR-T"

Evaluation:
  python job_search.py --evaluate               # Scrape + evaluate new jobs (default: last 1 day)
  python job_search.py --evaluate-only           # Evaluate without scraping first
  python job_search.py --eval-since "2026-02-19 06:30"
  python job_search.py --eval-days 3
  python job_search.py --eval-all                # All unevaluated
  python job_search.py --eval-prefilter-only     # Stage 1 only, no API calls
  python job_search.py --eval-dry-run            # Show count + cost estimate
  python job_search.py --eval-summary            # Show evaluation stats
  python job_search.py --eval-export results.csv # Export evaluations to CSV
  python job_search.py --eval-min-score 60       # Filter to 60+ when exporting
  python job_search.py --re-evaluate             # Force re-evaluation of already-scored jobs
        """,
    )
    # Search args
    parser.add_argument("--days", type=int, help="Number of days to look back (default: from config)")
    parser.add_argument("--terms", nargs="+", help="Search terms (overrides config)")
    parser.add_argument("--extra-terms", nargs="+", help="Additional search terms (appended to config terms)")
    parser.add_argument("--location", type=str, help="Location filter (default: 'United States')")
    parser.add_argument("--sites", nargs="+", help="Job sites to search (e.g. indeed linkedin)")
    parser.add_argument("--fetch-descriptions", action="store_true", default=None,
                        help="Fetch full job descriptions (slower)")
    parser.add_argument("--reprocess", action="store_true",
                        help="Re-run filtering and dedup from raw data (skips scraping)")
    parser.add_argument("--web", action="store_true", help="Open dashboard (no new scrape)")

    # Evaluation args
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate new jobs after scraping (default: last N days from config)")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Evaluate jobs without running scraper first")
    parser.add_argument("--eval-since", type=str,
                        help="Evaluate jobs posted since this datetime (e.g. '2026-02-19 06:30')")
    parser.add_argument("--eval-days", type=int,
                        help="Evaluate jobs from the last N days")
    parser.add_argument("--eval-all", action="store_true",
                        help="Evaluate all unevaluated jobs in the CSV")
    parser.add_argument("--eval-prefilter-only", action="store_true",
                        help="Run Stage 1 pre-filter only (no API calls)")
    parser.add_argument("--eval-dry-run", action="store_true",
                        help="Show count and cost estimate without calling API")
    parser.add_argument("--eval-summary", action="store_true",
                        help="Show evaluation statistics")
    parser.add_argument("--eval-export", type=str, metavar="FILE",
                        help="Export evaluation results to CSV file")
    parser.add_argument("--eval-min-score", type=int, default=0,
                        help="Minimum fit score filter (for export, default: 0)")
    parser.add_argument("--re-evaluate", action="store_true",
                        help="Force re-evaluation of already-scored jobs")

    return parser.parse_args()


def launch_dashboard():
    """Launch the Streamlit dashboard."""
    dashboard_path = PROJECT_ROOT / "src" / "dashboard.py"
    logger.info(f"Launching dashboard at http://localhost:8501")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.headless", "true",
    ])


def cli_progress_callback(scraper_name, status, count=0):
    """Print-based progress callback for CLI use."""
    if status == "starting":
        logger.info(f"  Starting {scraper_name}...")
    elif status == "done":
        logger.info(f"  {scraper_name} done — {count} results")


def load_jobs_csv(config) -> pd.DataFrame:
    """Load the master jobs CSV."""
    from src.exporter import get_master_path
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


def run_evaluation(args, config):
    """Run the evaluation pipeline based on CLI args."""
    from src.eval_persistence import EvaluationStore
    from src.evaluator import run_evaluation_pipeline, estimate_cost

    store = EvaluationStore(config.evaluation)

    # --eval-summary: just show stats and exit
    if args.eval_summary:
        summary = store.summary()
        if summary["total"] == 0:
            logger.info("No evaluations found. Run --evaluate first.")
            return
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total evaluated:    {summary['total']}")
        print(f"Avg score (API):    {summary['avg_score']}")
        print(f"Max score:          {summary['max_score']}")
        print(f"\nBuckets:")
        for bucket, count in summary["buckets"].items():
            print(f"  {bucket:20s} {count}")
        print(f"\nRecommendations:")
        for rec, count in summary["recommendations"].items():
            print(f"  {rec:20s} {count}")
        print(f"\nTokens used:")
        print(f"  Input:  {summary['total_input_tokens']:,}")
        print(f"  Output: {summary['total_output_tokens']:,}")
        meta = summary.get("metadata", {})
        if meta.get("last_evaluation_run"):
            print(f"\nLast run: {meta['last_evaluation_run']}")
        print(f"{'='*60}\n")
        return

    # Load jobs
    df = load_jobs_csv(config)
    if df.empty:
        return

    # --eval-export: export and exit
    if args.eval_export:
        path = store.export_csv(df, args.eval_export, args.eval_min_score)
        logger.info(f"Exported to {path}")
        return

    # Filter by time
    filtered = filter_jobs_by_time(
        df,
        eval_since=args.eval_since,
        eval_days=args.eval_days,
        eval_all=args.eval_all,
        default_days=config.evaluation.default_days,
    )

    if filtered.empty:
        logger.warning("No jobs match the time filter.")
        return

    # --eval-dry-run: show estimate and exit
    if args.eval_dry_run:
        est = estimate_cost(filtered, config.evaluation, store)
        print(f"\n{'='*60}")
        print(f"EVALUATION DRY RUN")
        print(f"{'='*60}")
        print(f"Total jobs in filter:   {est['total_jobs']}")
        print(f"Pre-filter skip:        {est['prefilter_skip']}")
        print(f"Already evaluated:      {est['already_evaluated']}")
        print(f"To evaluate (API):      {est['to_evaluate']}")
        print(f"Est. input tokens:      {est['estimated_input_tokens']:,}")
        print(f"Est. output tokens:     {est['estimated_output_tokens']:,}")
        print(f"Est. cost (USD):        ${est['estimated_cost_usd']:.4f}")
        print(f"{'='*60}\n")
        return

    # Run pipeline
    def progress(completed, total):
        logger.info(f"  Evaluated {completed}/{total} jobs")

    summary = run_evaluation_pipeline(
        jobs_df=filtered,
        config=config.evaluation,
        store=store,
        prefilter_only=args.eval_prefilter_only,
        re_evaluate=args.re_evaluate,
        progress_callback=progress,
    )

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total jobs processed:   {summary['total']}")
    print(f"Pre-filter skipped:     {summary['prefilter_skipped']}")
    print(f"Boosted (priority):     {summary['boosted']}")
    print(f"Descriptions fetched:   {summary.get('descriptions_fetched', 0)}")
    print(f"Already evaluated:      {summary['already_evaluated']}")
    print(f"Evaluated (API):        {summary['evaluated']}")
    print(f"{'='*60}\n")


def main():
    args = parse_args()

    # Build config with CLI overrides
    cli_overrides = {}
    if args.terms:
        cli_overrides["terms"] = args.terms
    if args.extra_terms:
        cli_overrides["extra_terms"] = args.extra_terms
    if args.days is not None:
        cli_overrides["days"] = args.days
    if args.location:
        cli_overrides["location"] = args.location
    if args.sites:
        cli_overrides["sites"] = args.sites
    if args.fetch_descriptions is not None:
        cli_overrides["fetch_descriptions"] = args.fetch_descriptions

    config = build_config(cli_overrides)

    # --web: just open dashboard with existing data
    if args.web:
        launch_dashboard()
        return

    # --eval-summary, --eval-export, --eval-dry-run: evaluation-only commands
    if args.eval_summary or args.eval_export or args.eval_dry_run:
        run_evaluation(args, config)
        return

    # --evaluate-only: evaluate without scraping
    if args.evaluate_only or args.eval_prefilter_only:
        run_evaluation(args, config)
        return

    # --reprocess: re-run filter/dedup from raw data (no scraping)
    if args.reprocess:
        logger.info("Reprocessing from raw data (skipping scrape)...")
        df = reprocess(config)
        if df.empty:
            logger.warning("No results after reprocessing.")
        else:
            logger.info(f"Reprocess complete! {len(df)} total jobs in master CSV.")
        return

    # Run search (scrapers run in parallel)
    logger.info(f"Search config: terms={config.search.terms}, days={config.search.days}, "
                f"sites={config.search.sites}, location={config.search.location}")
    df = run_search(config, progress_callback=cli_progress_callback)

    if df.empty:
        logger.warning("No results found. Try broadening your search terms or increasing --days.")
        return

    logger.info(f"Search complete! {len(df)} total jobs in master CSV.")

    # --evaluate: run evaluation after search
    if args.evaluate:
        run_evaluation(args, config)


if __name__ == "__main__":
    main()
```

---

## File: config.yaml

```yaml
# Pharma/Biotech Job Search Configuration
# Tailored for: CGT, Bioanalytical, Molecular Biology, Oncology specialist

search:
  terms:
    # Broad terms (cast a wide net, discipline filter narrows results)
    - "scientist pharma"
    - "scientist biotech"
    - "biologist pharma"
    # Specific roles
    - "cell gene therapy scientist"
    - "bioanalytical scientist"
    - "molecular biology scientist"
    - "oncology scientist"
    - "method validation scientist"
    - "CAR-T scientist"
    - "gene therapy manufacturing"
    - "DMPK scientist"
    - "flow cytometry scientist"
    # Scientist (targeted variants)
    - "research scientist"
    - "senior scientist"
    - "principal scientist"
    # Director-level roles
    - "associate director bioanalytical"
    - "director of science pharma"
    - "director of science biotech"
    - "bioanalytical director"
    - "director molecular biology"
    - "director cell gene therapy"
    - "director biotech"
    - "director pharma"
    # New categories
    - "translational research"
    - "precision medicine"
    - "analytical development"

  sites:
    - indeed
    - linkedin
    # zip_recruiter, glassdoor, google removed — anti-bot blocks return 0 results via JobSpy
  days: 7
  results_per_site: 100
  delay_between_searches: 5
  location: "United States"
  country_indeed: "USA"
  fetch_descriptions: false

  # Synonym expansion: each key auto-expands into its aliases during search
  # e.g., if "cell gene therapy" is a search term, its synonyms are also searched
  synonyms:
    "cell gene therapy":
      - "CGT scientist"
      - "CAR-T scientist"
      - "gene therapy"
      - "cell therapy"
    "bioanalytical":
      - "bioanalysis"
      - "ligand binding assay"
      - "PK/PD"
      - "pharmacokinetics"
    "molecular biology":
      - "molecular biologist"
      - "qPCR"
      - "PCR scientist"
      - "genomics"
      - "ddPCR"
      - "organoid"
      - "primary cell isolation"
    "oncology":
      - "immuno-oncology"
      - "tumor biology"
      - "cancer research"
      - "hematology"
    "method validation":
      - "assay validation"
      - "analytical method"
      - "GLP scientist"
    "DMPK":
      - "drug metabolism"
      - "pharmacokinetics"
      - "ADME"
    "flow cytometry":
      - "FACS"
      - "flow cytometry scientist"
    "research scientist":
      - "senior scientist"
      - "principal scientist"
    "translational research":
      - "translational medicine"
      - "translational scientist"
      - "clinical development scientist"
    "precision medicine":
      - "personalized medicine"
      - "biomarker"
      - "companion diagnostics"
      - "targeted therapy"
    "analytical development":
      - "analytical scientist"
      - "method development"
      - "analytical chemist"
      - "stability scientist"
    "director":
      - "associate director"
      - "senior director"
      - "director of science"
      - "head of"
      - "director of bioanalytical"
      - "director of molecular biology"

  # Post-scrape discipline filter: keep only biology/medicine roles
  # Titles/descriptions must match at least one include keyword
  filter_include:
    - biolog
    - bioanalyt
    - molecular
    - cell therapy
    - cell line
    - cell culture
    - cell biology
    - stem cell
    - CAR-T
    - gene therapy
    - gene editing
    - genetic
    - genomic
    - oncolog
    - cancer
    - immuno
    - pharma
    - biotech
    - qpcr
    - pcr
    - assay
    - potency
    - biodistribution
    - viral vector
    - flow cytometry
    - translational
    - preclinical
    - clinical research
    - clinical development
    - clinical trial
    - clinical scientist
    - clinical study
    - therapeutic
    - drug discovery
    - GLP
    - GMP
    - biomanufactur
    - GMP manufactur
    - manufacturing scientist
    - manufacturing director
    - quality control
    - method validation
    - assay validation
    - validation scientist
    - analytical validation
    - regulatory
    - scientist
    - science
    - associate director
    - translational
    - precision medicine
    - personalized medicine
    - biomarker
    - companion diagnostic
    - targeted therap
    - analytical
    - stability
    - method development
    - ddpcr
    - organoid
    - primary cell

  # Exclude non-biology disciplines and irrelevant roles
  filter_exclude:
    # Engineering (non-bio)
    - mechanical engineer
    - electrical engineer
    - software engineer
    - data engineer
    - civil engineer
    - environmental
    - chemical engineer
    - materials scientist
    - physics
    - physicist
    - geolog
    - petroleum
    - mining
    - aerospace
    - computational
    - quantum
    - fermentation
    - LC-MS
    # Tech company "scientist" roles (not pharma research)
    - applied scientist
    - applied science
    - robotics
    - sponsored products
    - product management
    - enablement
    - payer relations
    - advertiser
    # Data Science, AI & Machine Learning
    - data scientist
    - data science
    - data analyst
    - machine learning
    - deep learning
    - artificial intelligence
    - AI engineer
    - AI scientist
    - NLP engineer
    - computer vision engineer
    - MLOps
    - LLM
    # Sales & Commercial
    - sales
    - territory manager
    - account executive
    - business development
    # Administrative & Operations
    - administrative assistant
    - executive assistant
    - office manager
    - receptionist
    - data entry
    - coordinator
    # IT & Engineering
    - IT specialist
    - database administrator
    - systems administrator
    - network engineer
    - help desk
    # HR & Recruiting
    - recruiter
    - talent acquisition
    - human resources
    - HR manager
    # Marketing & Communications
    - marketing manager
    - brand manager
    - communications specialist
    - copywriter
    # Supply Chain & Logistics
    - supply chain
    - logistics
    - procurement
    - purchasing
    - inventory
    # Finance & Accounting
    - accountant
    - financial analyst
    - accounts payable
    - bookkeeper
    - controller
    - financial advisor
    - insurance agent
    # Facilities & Maintenance
    - maintenance technician
    - facilities manager
    - HVAC
    - plumber
    - electrician
    - janitor
    - custodian
    - housekeeper
    # Pharmacy (dispensing, not research)
    - pharmacist
    - pharmacy technician
    - pharmacy manager
    - dispensing
    # Teaching & Academic (not research)
    - tutor
    - lecturer
    - instructor
    - adjunct professor
    - assistant professor
    - associate professor
    - teaching assistant
    - teacher
    # Healthcare / Nursing / Clinical care (not research)
    - nurse
    - " RN "
    - " RN-"
    - ",RN"
    - "(RN)"
    - "RN "
    - " RN,"
    - physician
    - oncologist
    - microbiologist
    - clinical partner
    - clinical operations
    - dental
    - dentist
    - veterinar
    - occupational therapist
    - physical therapist
    - respiratory therapist
    - speech therapist
    - behavioral therapist
    - licensed therapist
    - social worker
    - counselor
    - paramedic
    - phlebotom
    - medical assistant
    - medical coder
    - medical biller
    - medical director
    - medical lab
    - medical science liaison
    - patient care
    - home health
    - hospice
    - pediatric
    - infusion
    - transplant
    - marrow
    - on-call
    - PRN
    - technologist
    - clinical lab
    - clinical laboratory
    - acute care
    - per diem
    - support tech
    # Internships & Trainee positions
    - postdoctoral
    - intern
    - co-op
    # Commercial / Business (not research)
    - commercial development
    - commercial director
    - commercial manager
    - project manager
    - program manager
    - operations manager
    - director of operations
    # Hospital & Healthcare administration (not research)
    - nursing
    - hospital
    - ICU
    - critical care
    - emergency services
    - emergency department
    - emergency medicine
    - case manager
    - case management
    - nutrition
    - dietary
    - food scientist
    - food safety
    - food science
    - beverage
    - college of nursing
    - wound care
    - perioperative
    - labor and delivery
    - NICU
    - chief nursing
    - bedside
    - inpatient
    - surgical services
    - rehabilitation
    - radiology
    - medical records
    - health information
    - infection control
    - utilization review
    - discharge planning
    - admissions director
    - urgent care
    # Legal & Creative (not research)
    - attorney
    - counsel
    - paralegal
    - creative director
    # Education administration (not research)
    - financial aid
    - genealogy
    # Non-pharma science & education
    - measurement science
    - social science
    - imaging science
    - recreation
    - school district
    # General labor & retail
    - delivery driver
    - fitness
    - personal trainer
    - general manager
    - car wash
    - franchise
    - cashier
    - barista
    - retail associate
    - store manager
    - warehouse
    - forklift
    - food service
    - restaurant
    - truck driver
    - CDL
    - real estate
    - customer service
    - account manager

usajobs:
  # Register at https://developer.usajobs.gov/ to get an API key
  # Set these values or use environment variables USAJOBS_API_KEY and USAJOBS_EMAIL
  api_key: ""  # Set via USAJOBS_API_KEY env var
  email: ""  # Set via USAJOBS_EMAIL env var
  keywords:
    - "cell gene therapy"
    - "molecular biology scientist"
    - "bioanalytical scientist"
    - "oncology research"
    - "scientist pharma"
    - "scientist biotech"
    - "CAR-T"
    - "DMPK"
    - "director bioanalytical"
    - "director of science"

adzuna:
  # Register at https://developer.adzuna.com/ to get app_id and app_key
  # Set these values or use environment variables ADZUNA_APP_ID and ADZUNA_APP_KEY
  # Free tier: 25 requests/min, 250 requests/day
  app_id: ""  # Set via ADZUNA_APP_ID env var
  app_key: ""  # Set via ADZUNA_APP_KEY env var
  keywords:
    - "cell gene therapy"
    - "molecular biology scientist"
    - "bioanalytical scientist"
    - "oncology research"
    - "scientist pharma"
    - "scientist biotech"
    - "CAR-T"
    - "DMPK"
    - "director bioanalytical"
    - "director of science"
  results_per_page: 50
  max_pages: 5

jooble:
  # Register at https://jooble.org/api/about to get an API key
  # Set this value or use environment variable JOOBLE_API_KEY
  api_key: ""
  keywords:
    - "cell gene therapy"
    - "molecular biology scientist"
    - "bioanalytical scientist"
    - "oncology research"
    - "scientist pharma"
    - "scientist biotech"
    - "CAR-T"
    - "DMPK"
    - "director bioanalytical"
    - "director of science"
  max_pages: 5

output:
  directory: "data"
  filename_prefix: "pharma_jobs"
  formats:
    - csv
    - excel

dashboard:
  port: 8501
  max_results: 2000

evaluation:
  # Anthropic API key — set here or via ANTHROPIC_API_KEY env var
  anthropic_api_key: "YOUR_ANTHROPIC_API_KEY"
  model: "claude-haiku-4-5-20251001"
  resume_profile: "data/resume_profile.json"
  evaluations_store: "data/evaluations.json"
  max_concurrent: 2
  delay_between_calls: 2.0
  default_days: 1
  description_max_chars: 6000
```

---

## File: requirements.txt

```
python-jobspy>=1.1.0
pandas>=2.0.0
openpyxl>=3.1.0
streamlit>=1.30.0
streamlit-aggrid>=1.0.0
pyyaml>=6.0
requests>=2.31.0
beautifulsoup4>=4.12.0
anthropic>=0.40.0
```

---

## File: BioTechNerd-Apache's Job Search.command

```bash
#!/usr/bin/env bash
# ============================================================
# BioTechNerd-Apache's Job Search — Dashboard Only (no scraping)
# Double-click to view your latest job search results in Chrome
# ============================================================

PROJECT_DIR="$HOME/Library/Mobile Documents/com~apple~CloudDocs/My Vibe Code/pharma-job-search"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
PORT=8501
URL="http://localhost:$PORT"

# Kill any existing Streamlit on this port
lsof -ti :$PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 1

echo "========================================"
echo "  BioTechNerd-Apache's Job Search Dashboard"
echo "========================================"
echo ""
echo "Loading latest data from: $PROJECT_DIR/data/"
echo ""

# Force iCloud to download data files before launching dashboard
# (prevents stale data when opening on a different Mac)
DATA_DIR="$PROJECT_DIR/data"
if [ -d "$DATA_DIR" ]; then
    echo "Syncing data files from iCloud..."
    for f in "$DATA_DIR"/pharma_jobs.csv "$DATA_DIR"/pharma_jobs.xlsx "$DATA_DIR"/pharma_jobs_raw.csv "$DATA_DIR"/reviewed.json; do
        if [ -e "$f" ]; then
            brctl download "$f" 2>/dev/null
        fi
    done
    # Wait up to 30 seconds for the main CSV to finish downloading
    for i in $(seq 1 30); do
        if [ -f "$DATA_DIR/pharma_jobs.csv" ] && ! brctl download "$DATA_DIR/pharma_jobs.csv" 2>&1 | grep -q "not evicted"; then
            # File is local — verify it's not still being written by checking stable size
            SIZE1=$(stat -f%z "$DATA_DIR/pharma_jobs.csv" 2>/dev/null || echo 0)
            sleep 1
            SIZE2=$(stat -f%z "$DATA_DIR/pharma_jobs.csv" 2>/dev/null || echo 0)
            if [ "$SIZE1" = "$SIZE2" ] && [ "$SIZE1" -gt 0 ]; then
                echo "Data files synced."
                break
            fi
        fi
        echo "  Waiting for iCloud sync... ($i/30)"
        sleep 1
    done
fi

# Start Streamlit dashboard only (NO scraping)
cd "$PROJECT_DIR"
"$PYTHON" -m streamlit run src/dashboard.py \
    --server.port $PORT \
    --server.headless true \
    --browser.gatherUsageStats false &

STREAMLIT_PID=$!

# Wait for server to be ready
echo "Starting dashboard..."
for i in $(seq 1 30); do
    if curl -s "$URL" > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

# Open in Chrome
if [ -d "/Applications/Google Chrome.app" ]; then
    open -a "Google Chrome" "$URL"
else
    open "$URL"
fi

echo ""
echo "Dashboard running at: $URL"
echo ""
echo "Click the '🔍 Run New Search' button in the sidebar to search for new jobs."
echo "Or run from Terminal:  cd \"$PROJECT_DIR\" && python3 job_search.py"
echo ""
echo "Press Ctrl+C to stop the dashboard."

wait $STREAMLIT_PID
```

---

## File: src/config.py

```python
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
    priority_results_per_site: int = 200
    priority_terms: List[str] = field(default_factory=list)
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
    for key in ("terms", "sites", "days", "results_per_site",
                "priority_results_per_site", "priority_terms",
                "delay_between_searches",
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
```

---

## File: src/aggregator.py

```python
"""Orchestrates scrapers, normalizes columns, merges, deduplicates, sorts."""

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

import pandas as pd

from .config import AppConfig, SearchConfig
from .scraper_jobspy import scrape_jobs, scrape_single_term
from .scraper_usajobs import scrape_usajobs
from .scraper_adzuna import scrape_adzuna
from .scraper_jooble import scrape_jooble
from .dedup import deduplicate

logger = logging.getLogger(__name__)

# Standard output columns
OUTPUT_COLUMNS = [
    "title", "company", "location", "state", "date_posted",
    "reposted_date", "days_since_posted", "source",
    "job_url", "job_url_direct", "salary_min", "salary_max", "is_remote", "job_type",
    "description", "eval_status",
]

# US state abbreviations for extraction
US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC",
}


def extract_state(location: str) -> str:
    """Extract US state abbreviation from a location string."""
    if not location or not isinstance(location, str):
        return ""
    # Look for 2-letter state codes (common pattern: "City, ST" or "City, ST 12345")
    match = re.search(r",\s*([A-Z]{2})\b", location)
    if match and match.group(1) in US_STATES:
        return match.group(1)
    # Try at end of string
    match = re.search(r"\b([A-Z]{2})\s*$", location)
    if match and match.group(1) in US_STATES:
        return match.group(1)
    return ""


def normalize_jobspy_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize JobSpy output to our standard schema."""
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    result = pd.DataFrame()

    # Map JobSpy columns to our schema
    col_map = {
        "title": "title",
        "company_name": "company",
        "company": "company",
        "location": "location",
        "date_posted": "date_posted",
        "site": "source",
        "job_url": "job_url",
        "job_url_direct": "job_url_direct",
        "min_amount": "salary_min",
        "max_amount": "salary_max",
        "is_remote": "is_remote",
        "job_type": "job_type",
        "description": "description",
    }

    for src_col, dst_col in col_map.items():
        if src_col in df.columns and dst_col not in result.columns:
            result[dst_col] = df[src_col]

    # Ensure all output columns exist
    for col in OUTPUT_COLUMNS:
        if col not in result.columns:
            result[col] = None

    # Extract state from location
    if "state" not in result.columns or result["state"].isna().all():
        result["state"] = result["location"].apply(extract_state)

    # Ensure Indeed/LinkedIn jobs use job-board URLs (not external employer URLs)
    if "id" in df.columns:
        for idx, row in df.iterrows():
            site = str(row.get("site", "")).lower()
            job_id = str(row.get("id", ""))
            if site == "indeed" and job_id.startswith("in-"):
                jk = job_id[3:]  # strip "in-" prefix
                result.at[idx, "job_url"] = f"https://www.indeed.com/viewjob?jk={jk}"
            elif site == "linkedin" and job_id.startswith("li-"):
                li_id = job_id[3:]  # strip "li-" prefix
                result.at[idx, "job_url"] = f"https://www.linkedin.com/jobs/view/{li_id}"

    return result[OUTPUT_COLUMNS]


def apply_discipline_filter(df: pd.DataFrame, search_config: SearchConfig) -> pd.DataFrame:
    """Filter results to biology/medicine disciplines using include/exclude keyword lists.

    Both include and exclude filters match against job TITLE only to avoid
    false positives from keywords appearing in job descriptions.
    """
    if df.empty:
        return df

    include_kw = search_config.filter_include
    exclude_kw = search_config.filter_exclude

    if not include_kw and not exclude_kw:
        return df

    # Title-only text for both include and exclude matching
    title_col = df["title"].fillna("").str.lower()

    pre_count = len(df)

    # Exclude: remove rows whose TITLE matches any exclude keyword
    if exclude_kw:
        exclude_mask = title_col.apply(
            lambda t: any(kw.lower() in t for kw in exclude_kw)
        )
        excluded = exclude_mask.sum()
        df = df[~exclude_mask]
        title_col = title_col[~exclude_mask]
        if excluded > 0:
            logger.info(f"Discipline filter: excluded {excluded} non-biology/medicine jobs")

    # Include: keep only rows where the TITLE matches at least one include keyword
    if include_kw:
        include_mask = title_col.apply(
            lambda t: any(kw.lower() in t for kw in include_kw)
        )
        filtered_out = (~include_mask).sum()
        df = df[include_mask]
        if filtered_out > 0:
            logger.info(f"Discipline filter: removed {filtered_out} jobs not matching biology/medicine keywords in title")

    logger.info(f"Discipline filter: {pre_count} -> {len(df)} jobs")
    return df.reset_index(drop=True)


def _normalize_api_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Normalize an API scraper DataFrame to standard schema."""
    if df.empty:
        return df
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    if source_name in ("Adzuna", "Jooble"):
        df["state"] = df["location"].apply(extract_state)
    return df[OUTPUT_COLUMNS]


def _get_raw_path(config: AppConfig):
    """Return path to the raw intermediate CSV file."""
    from .config import PROJECT_ROOT
    raw_dir = PROJECT_ROOT / config.output.directory
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir / f"{config.output.filename_prefix}_raw.csv"


def run_search(config: AppConfig, progress_callback=None) -> pd.DataFrame:
    """Run all scrapers via a shared work queue. All 5 workers stay busy.

    API scrapers (USAJobs, Adzuna, Jooble) run as single tasks and finish fast.
    JobSpy scrapers (Indeed, LinkedIn) are split into individual (site, term) tasks
    so that all workers can share the load once API tasks are done.

    Per-site semaphores limit concurrency to 2 workers per site.
    Per-site delay tracking ensures delay_between_searches between consecutive
    requests to the same site.

    Args:
        config: Application configuration.
        progress_callback: Optional callable(scraper_name, status, count=0).
            status is "starting" or "done".
    """
    from .exporter import merge_and_export_csv

    csv_lock = threading.Lock()
    raw_lock = threading.Lock()
    total_saved = 0
    raw_path = _get_raw_path(config)

    # Per-site concurrency control: max 2 concurrent workers per site
    site_semaphores = {
        "indeed": threading.Semaphore(2),
        "linkedin": threading.Semaphore(2),
    }

    # Per-site delay tracking: last request timestamp per site
    site_last_request = {}
    site_delay_lock = threading.Lock()
    delay_seconds = config.search.delay_between_searches

    # Track per-source result counts for progress callback
    source_counts = {}
    source_counts_lock = threading.Lock()

    # Clear old raw file at the start of a new search
    if raw_path.exists():
        raw_path.unlink()

    def _enforce_site_delay(site: str):
        """Wait if needed to respect delay_between_searches for a site."""
        with site_delay_lock:
            last = site_last_request.get(site, 0)
            elapsed = time.time() - last
            wait_time = delay_seconds - elapsed
        if wait_time > 0:
            logger.info(f"Rate limiting: waiting {wait_time:.1f}s for {site}...")
            time.sleep(wait_time)
        with site_delay_lock:
            site_last_request[site] = time.time()

    def _save_results(source_name: str, df: pd.DataFrame):
        """Save raw + filtered results (thread-safe). Returns count saved."""
        nonlocal total_saved
        if df.empty:
            return 0

        # Append raw data
        with raw_lock:
            header = not raw_path.exists()
            df.to_csv(raw_path, mode="a", index=False, header=header)

        # Filter and save to master CSV
        filtered = apply_discipline_filter(df, config.search)
        if not filtered.empty:
            with csv_lock:
                merge_and_export_csv(filtered, config)
            total_saved += len(filtered)
            logger.info(f"{source_name}: saved {len(filtered)} jobs to master CSV")

        return len(df)

    def _run_api_scraper(name, scraper_func, *args):
        """Run an API scraper (USAJobs/Adzuna/Jooble) as a single task."""
        try:
            df = scraper_func(*args)
            df = _normalize_api_df(df, name)
            count = _save_results(name, df)
            logger.info(f"{name} returned {count} results")
            return (name, count)
        except Exception as e:
            logger.error(f"{name} scraper failed: {e}")
            return (name, 0)

    def _run_jobspy_term(site: str, term: str):
        """Run a single (site, term) JobSpy scrape with concurrency + delay control."""
        sem = site_semaphores.get(site)
        if sem:
            sem.acquire()
        try:
            _enforce_site_delay(site)
            df = scrape_single_term(config.search, site, term)
            if not df.empty:
                df = normalize_jobspy_df(df)
                count = _save_results(site.capitalize(), df)
            else:
                count = 0
            return (site, term, count)
        except Exception as e:
            logger.warning(f"Error scraping {site} for '{term}': {e}")
            return (site, term, 0)
        finally:
            if sem:
                sem.release()

    # Build the work queue
    # 1. API scrapers (fast — these go first)
    api_tasks = []
    api_tasks.append(("USAJobs", scrape_usajobs, config.usajobs, config.search.days))
    api_tasks.append(("Adzuna", scrape_adzuna, config.adzuna, config.search.days))
    api_tasks.append(("Jooble", scrape_jooble, config.jooble, config.search.days))

    # 2. JobSpy (site, term) pairs — interleaved so both sites get worked on simultaneously
    jobspy_sites = [s for s in config.search.sites if s in ("indeed", "linkedin")]
    terms = config.search.terms
    jobspy_tasks = []
    for i, term in enumerate(terms):
        for site in jobspy_sites:
            jobspy_tasks.append((site, term))

    total_tasks = len(api_tasks) + len(jobspy_tasks)
    logger.info(f"=== Work queue: {len(api_tasks)} API tasks + {len(jobspy_tasks)} JobSpy tasks ({total_tasks} total) ===")

    # Notify all sources starting
    all_sources = ["Indeed", "LinkedIn", "USAJobs", "Adzuna", "Jooble"]
    if progress_callback:
        for name in all_sources:
            progress_callback(name, "starting", count=0)

    # Track which sources have reported done
    source_done_counts = {s: 0 for s in all_sources}
    # Expected counts: API scrapers = 1 task each, JobSpy = number of terms per site
    source_expected = {}
    for name in ["USAJobs", "Adzuna", "Jooble"]:
        source_expected[name] = 1
    for site in jobspy_sites:
        source_expected[site.capitalize()] = len(terms)

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_label = {}

        # Submit API tasks first (they finish fast)
        for name, func, *args in api_tasks:
            future = executor.submit(_run_api_scraper, name, func, *args)
            future_to_label[future] = ("api", name)

        # Submit all JobSpy (site, term) tasks
        for site, term in jobspy_tasks:
            future = executor.submit(_run_jobspy_term, site, term)
            future_to_label[future] = ("jobspy", site.capitalize())

        for future in as_completed(future_to_label):
            task_type, source_name = future_to_label[future]
            try:
                result = future.result()
                if task_type == "api":
                    _, count = result
                else:
                    _, _, count = result

                # Track per-source completion for progress callback
                source_done_counts[source_name] = source_done_counts.get(source_name, 0) + 1
                with source_counts_lock:
                    source_counts[source_name] = source_counts.get(source_name, 0) + count

                expected = source_expected.get(source_name, 1)
                if source_done_counts[source_name] >= expected and progress_callback:
                    total_for_source = source_counts.get(source_name, 0)
                    progress_callback(source_name, "done", count=total_for_source)

            except Exception as e:
                logger.error(f"{source_name} task failed: {e}")
                source_done_counts[source_name] = source_done_counts.get(source_name, 0) + 1
                expected = source_expected.get(source_name, 1)
                if source_done_counts[source_name] >= expected and progress_callback:
                    progress_callback(source_name, "done", count=0)

    if total_saved == 0:
        logger.warning("No results from any source.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Load the final master CSV to return
    from .exporter import get_master_path
    master_path = get_master_path(config.output, "csv")
    if master_path.exists():
        final = pd.read_csv(master_path, parse_dates=["date_posted"])
        logger.info(f"Final master CSV: {len(final)} total jobs")
        return final

    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def reprocess(config: AppConfig) -> pd.DataFrame:
    """Re-run filtering and dedup from the raw intermediate file without re-scraping.

    Reads the raw normalized data saved by run_search, applies discipline filtering
    and deduplication, and writes a fresh master CSV + Excel.
    """
    from .exporter import get_master_path
    from .config import PROJECT_ROOT

    raw_path = _get_raw_path(config)
    if not raw_path.exists():
        logger.error(f"No raw data file found at {raw_path}. Run a search first.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    logger.info(f"Loading raw data from {raw_path.name}...")
    raw_df = pd.read_csv(raw_path, parse_dates=["date_posted"])
    logger.info(f"Loaded {len(raw_df)} raw rows")

    # Apply discipline filter
    filtered = apply_discipline_filter(raw_df, config.search)
    logger.info(f"After discipline filter: {len(filtered)} rows")

    if filtered.empty:
        logger.warning("No results after filtering.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Deduplicate
    filtered = deduplicate(filtered)
    logger.info(f"After dedup: {len(filtered)} rows")

    # Sort by date descending
    if "date_posted" in filtered.columns:
        filtered["date_posted"] = pd.to_datetime(filtered["date_posted"], errors="coerce")
        filtered = filtered.sort_values("date_posted", ascending=False, na_position="last")
        filtered = filtered.reset_index(drop=True)

    # Recalculate days_since_posted
    if "date_posted" in filtered.columns:
        today = pd.Timestamp.now().normalize()
        filtered["days_since_posted"] = (today - filtered["date_posted"]).dt.days

    # Ensure reposted_date column exists
    if "reposted_date" not in filtered.columns:
        filtered["reposted_date"] = ""

    # Preserve eval_status from existing master CSV during reprocess
    master_path = get_master_path(config.output, "csv")
    if master_path.exists():
        try:
            old_master = pd.read_csv(master_path, usecols=["job_url", "eval_status"])
            old_status = old_master[old_master["eval_status"].fillna("").astype(str).str.strip() != ""]
            if not old_status.empty:
                status_map = dict(zip(old_status["job_url"], old_status["eval_status"]))
                if "eval_status" not in filtered.columns:
                    filtered["eval_status"] = ""
                filtered["eval_status"] = filtered["job_url"].map(status_map).fillna("")
        except Exception:
            pass

    # Ensure eval_status column exists
    if "eval_status" not in filtered.columns:
        filtered["eval_status"] = ""
    filtered["eval_status"] = filtered["eval_status"].fillna("")

    # Save master CSV
    filtered.to_csv(master_path, index=False)
    logger.info(f"Master CSV saved: {master_path} ({len(filtered)} rows)")

    # Save Excel
    from .exporter import _export_excel
    _export_excel(filtered, config.output)

    return filtered
```

---

## File: src/dedup.py

```python
"""Deduplication: exact URL match + fuzzy (company + title + state) match
   + cross-source (title + fuzzy company) match.
   Tracks reposted dates when the same job appears with different dates.
   Reviewed jobs are always preferred when choosing which duplicate to keep."""

import json
import logging
import re
from pathlib import Path
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

import pandas as pd

logger = logging.getLogger(__name__)


def load_reviewed_urls() -> set:
    """Load the set of job URLs that have been marked as reviewed."""
    from .config import PROJECT_ROOT
    reviewed_path = PROJECT_ROOT / "data" / "reviewed.json"
    if reviewed_path.exists():
        try:
            with open(reviewed_path, "r") as f:
                return set(json.load(f).keys())
        except (json.JSONDecodeError, IOError):
            return set()
    return set()

# Common company name suffixes/words to strip for fuzzy matching
COMPANY_STOP_WORDS = {
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
    "co", "company", "companies", "group", "holdings", "holding",
    "the", "and", "of", "for",
    "plc", "sa", "nv", "ag", "gmbh", "se", "lp", "llp",
    "us", "usa", "global", "international", "intl",
    "pharmaceuticals", "pharmaceutical", "pharma",
    "therapeutics", "biotherapeutics", "biosciences", "bioscience",
    "biotechnology", "biotech", "sciences", "science",
    "healthcare", "health", "medical", "laboratories", "labs", "lab",
}

# Known job boards, recruiters, and aggregator sites that repost employer jobs.
# When one company in a title-match pair is a known board, treat as duplicate.
KNOWN_JOB_BOARDS = {
    # General job boards / aggregators
    "biospace", "biopharmguy", "medreps", "ladders", "dice", "monster",
    "careerbuilder", "simplyhired", "snagajob", "ziprecruiter", "glassdoor",
    "lensa", "talent", "talentify", "jobvite", "lever", "greenhouse",
    "smartrecruiters", "workday", "icims", "taleo", "brassring",
    "recruitics", "appcast", "neuvoo", "jooble", "adzuna", "getwork",
    "linkup", "jobrapido", "whatjobs", "pandologic", "joveo",
    # Science / pharma specific boards
    "nature jobs", "naturejobs", "science careers", "sciencecareers",
    "aaas", "asbmb", "acs", "abrf", "higheredjobs", "chronicle vitae",
    "academickeys", "academiccareers", "insidehighered",
    # Staffing / recruiting agencies
    "robert half", "roberthalf", "randstad", "kelly services", "kellyservices",
    "hays", "michael page", "michaelpage", "adecco", "manpower",
    "aerotek", "insight global", "insightglobal", "kforce", "teksystems",
    "actalent", "yoh", "prium", "proclinical",
}

def _is_job_board(company: str) -> bool:
    """Check if a company name matches a known job board or recruiter."""
    if not company or not isinstance(company, str):
        return False
    name_lower = company.lower().strip()
    # Check exact and normalized matches
    name_norm = re.sub(r"[^a-z0-9\s]", " ", name_lower).strip()
    name_joined = name_norm.replace(" ", "")
    for board in KNOWN_JOB_BOARDS:
        if board in name_lower or board in name_norm or board == name_joined:
            return True
    return False


# URL parameters to strip (tracking/session params)
STRIP_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "ref", "refId", "trackingId", "trk", "tk",
    "currentJobId", "eBP", "recommendedFlavor", "position", "pageNum",
}


def normalize_url(url: str) -> str:
    """Strip tracking parameters and fragments from a URL."""
    if not url or not isinstance(url, str):
        return ""
    try:
        parsed = urlparse(url)
        params = parse_qs(parsed.query, keep_blank_values=False)
        filtered = {k: v for k, v in params.items() if k not in STRIP_PARAMS}
        clean_query = urlencode(filtered, doseq=True)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, clean_query, ""))
    except Exception:
        return url


def normalize_text(text: str) -> str:
    """Lowercase, strip whitespace and non-alphanumeric chars for comparison."""
    if not text or not isinstance(text, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", text.lower().strip())


def normalize_company(name: str) -> str:
    """Aggressively normalize a company name for cross-source matching.

    Strips punctuation, common suffixes (Inc, LLC, Corp, etc.),
    and industry words (Pharmaceuticals, Therapeutics, etc.).
    """
    if not name or not isinstance(name, str):
        return ""
    # Lowercase, replace punctuation with spaces
    cleaned = re.sub(r"[^a-z0-9\s]", " ", name.lower().strip())
    # Remove stop words
    words = [w for w in cleaned.split() if w and w not in COMPANY_STOP_WORDS]
    return "".join(words)


def companies_match(a: str, b: str) -> bool:
    """Check if two company names likely refer to the same company.

    Returns True if:
    1. Normalized names are equal, or one is a substring of the other (min 4 chars).
       E.g. "Eli Lilly and Company" -> "elililly" contains "lilly" <- "Lilly".
    2. Either company is a known job board / recruiter (they repost employer jobs).
       E.g. "BioSpace" is a job board, so BioSpace + any employer = match.
    """
    a_norm = normalize_company(a)
    b_norm = normalize_company(b)
    if not a_norm or not b_norm:
        return False
    if a_norm == b_norm:
        return True
    # Substring check: shorter must be >= 4 chars to avoid false positives
    shorter, longer = (a_norm, b_norm) if len(a_norm) <= len(b_norm) else (b_norm, a_norm)
    if len(shorter) >= 4 and shorter in longer:
        return True
    # If either is a known job board, treat as duplicate (boards repost employer jobs)
    if _is_job_board(a) or _is_job_board(b):
        return True
    return False


def data_richness_score(row: pd.Series) -> int:
    """Score how much useful data a row contains (higher = richer)."""
    score = 0
    for col in ["description", "salary_min", "salary_max", "job_type", "is_remote", "location"]:
        if col in row.index:
            val = row[col]
            if pd.notna(val) and val != "" and val is not False:
                score += 1
    if "description" in row.index and isinstance(row["description"], str):
        score += min(len(row["description"]) // 100, 5)
    return score


def _collect_dates_for_group(group: pd.DataFrame) -> str:
    """Collect all unique dates from a group of duplicate rows, return comma-separated repost dates."""
    dates = group["date_posted"].dropna().unique()
    date_strs = set()
    for d in dates:
        try:
            date_strs.add(str(pd.Timestamp(d).date()))
        except Exception:
            continue
    return date_strs


def _dedup_with_dates(df: pd.DataFrame, key_col: str, has_key_mask: pd.Series) -> pd.DataFrame:
    """Deduplicate on key_col, collecting all date_posted values into _all_dates set."""
    keyed = df[has_key_mask].copy()
    no_key = df[~has_key_mask].copy()

    if keyed.empty:
        return df

    # Collect all dates per group before deduplication
    date_map = {}
    for key, group in keyed.groupby(key_col):
        date_map[key] = _collect_dates_for_group(group)

    # Keep richest row per key
    keyed = keyed.sort_values("_richness", ascending=False)
    deduped = keyed.drop_duplicates(subset=[key_col], keep="first").copy()

    # Merge collected dates back
    deduped["_all_dates"] = deduped[key_col].map(date_map)

    # For no_key rows, initialize _all_dates from their own date
    if "_all_dates" not in no_key.columns:
        no_key["_all_dates"] = no_key["date_posted"].apply(
            lambda d: {str(pd.Timestamp(d).date())} if pd.notna(d) else set()
        )

    result = pd.concat([deduped, no_key], ignore_index=True)
    return result


def deduplicate(df: pd.DataFrame, reviewed_urls: set | None = None) -> pd.DataFrame:
    """Two-layer deduplication: exact URL match, then fuzzy (company+title+state).
    Collects all date_posted values from duplicates to detect reposts.
    Reviewed jobs are always preferred when choosing which duplicate to keep."""
    if df.empty:
        return df

    if reviewed_urls is None:
        reviewed_urls = load_reviewed_urls()

    initial_count = len(df)
    df = df.copy()
    df["_richness"] = df.apply(data_richness_score, axis=1)

    # Boost richness for reviewed jobs so they always win in dedup
    if reviewed_urls and "job_url" in df.columns:
        reviewed_mask = df["job_url"].isin(reviewed_urls)
        df.loc[reviewed_mask, "_richness"] += 1000

    # Initialize _all_dates from each row's own date
    df["_all_dates"] = df["date_posted"].apply(
        lambda d: {str(pd.Timestamp(d).date())} if pd.notna(d) else set()
    )

    # Layer 1: Exact URL dedup
    if "job_url" in df.columns:
        df["_norm_url"] = df["job_url"].apply(normalize_url)
        has_url = df["_norm_url"] != ""

        # Collect dates before dedup
        if has_url.any():
            for key, group in df[has_url].groupby("_norm_url"):
                combined_dates = set()
                for dates_set in group["_all_dates"]:
                    combined_dates.update(dates_set)
                df.loc[group.index, "_all_dates"] = [combined_dates] * len(group)

            df = df.sort_values("_richness", ascending=False)
            has_url = df["_norm_url"] != ""
            deduped_urls = df.loc[has_url].drop_duplicates(subset=["_norm_url"], keep="first")
            no_url = df.loc[~has_url]
            df = pd.concat([deduped_urls, no_url], ignore_index=True)

        url_dedup_count = initial_count - len(df)
        if url_dedup_count > 0:
            logger.info(f"URL dedup removed {url_dedup_count} duplicates")

    # Layer 2: Fuzzy dedup on (company + title + state)
    pre_fuzzy = len(df)
    df["_fuzzy_key"] = (
        df.get("company", pd.Series(dtype=str)).fillna("").apply(normalize_text) + "|" +
        df.get("title", pd.Series(dtype=str)).fillna("").apply(normalize_text) + "|" +
        df.get("state", pd.Series(dtype=str)).fillna("").apply(normalize_text)
    )

    has_key = df["_fuzzy_key"] != "||"

    if has_key.any():
        # Collect dates before dedup
        for key, group in df[has_key].groupby("_fuzzy_key"):
            combined_dates = set()
            for dates_set in group["_all_dates"]:
                combined_dates.update(dates_set)
            df.loc[group.index, "_all_dates"] = [combined_dates] * len(group)

        df = df.sort_values("_richness", ascending=False)
        has_key = df["_fuzzy_key"] != "||"
        deduped_fuzzy = df.loc[has_key].drop_duplicates(subset=["_fuzzy_key"], keep="first")
        no_key = df.loc[~has_key]
        df = pd.concat([deduped_fuzzy, no_key], ignore_index=True)

    fuzzy_dedup_count = pre_fuzzy - len(df)
    if fuzzy_dedup_count > 0:
        logger.info(f"Fuzzy dedup removed {fuzzy_dedup_count} duplicates")

    # Layer 3: Cross-source dedup (same title + similar company, across different sources)
    pre_cross = len(df)
    df["_norm_title"] = df.get("title", pd.Series(dtype=str)).fillna("").apply(normalize_text)

    # Group by normalized title, then check company similarity within each group
    to_drop = set()
    for title_key, group in df.groupby("_norm_title"):
        if len(group) < 2 or title_key == "":
            continue
        indices = group.index.tolist()
        for i in range(len(indices)):
            if indices[i] in to_drop:
                continue
            for j in range(i + 1, len(indices)):
                if indices[j] in to_drop:
                    continue
                row_i = df.loc[indices[i]]
                row_j = df.loc[indices[j]]
                # Check company similarity
                if not companies_match(str(row_i.get("company", "")), str(row_j.get("company", ""))):
                    continue
                # Match found — merge dates, keep richer row
                dates_i = row_i.get("_all_dates", set())
                dates_j = row_j.get("_all_dates", set())
                combined_dates = (dates_i or set()) | (dates_j or set())
                rich_i = row_i.get("_richness", 0)
                rich_j = row_j.get("_richness", 0)
                if rich_j > rich_i:
                    keep_idx, drop_idx = indices[j], indices[i]
                else:
                    keep_idx, drop_idx = indices[i], indices[j]
                df.at[keep_idx, "_all_dates"] = combined_dates
                to_drop.add(drop_idx)

    if to_drop:
        df = df.drop(index=to_drop).reset_index(drop=True)

    cross_dedup_count = pre_cross - len(df)
    if cross_dedup_count > 0:
        logger.info(f"Cross-source dedup removed {cross_dedup_count} duplicates")

    # Build reposted_date column from _all_dates
    # Original date = date_posted (earliest), repost dates = all other dates
    def extract_repost_dates(row):
        all_dates = row.get("_all_dates", set())
        if not all_dates or len(all_dates) <= 1:
            return ""
        original = str(pd.Timestamp(row["date_posted"]).date()) if pd.notna(row["date_posted"]) else None
        repost_dates = sorted(d for d in all_dates if d != original)
        return ", ".join(repost_dates) if repost_dates else ""

    df["reposted_date"] = df.apply(extract_repost_dates, axis=1)

    repost_count = (df["reposted_date"] != "").sum()
    if repost_count > 0:
        logger.info(f"Detected {repost_count} reposted jobs")

    # Clean up temp columns
    df = df.drop(columns=["_richness", "_norm_url", "_fuzzy_key", "_norm_title", "_all_dates"], errors="ignore")

    total_removed = initial_count - len(df)
    logger.info(f"Deduplication: {initial_count} -> {len(df)} ({total_removed} removed)")
    return df


def merge_historical_reposts(df: pd.DataFrame, history_dir, filename_prefix: str) -> pd.DataFrame:
    """Compare current results against previous CSV files to detect cross-run reposts."""
    from pathlib import Path

    history_path = Path(history_dir)
    if not history_path.exists():
        return df

    csv_files = sorted(history_path.glob(f"{filename_prefix}_*.csv"), reverse=True)
    if not csv_files:
        return df

    # Load previous runs (skip the most recent since that's probably the current one being written)
    historical_dates = {}  # fuzzy_key -> set of date strings
    for csv_file in csv_files[:5]:  # Check last 5 files max
        try:
            hist = pd.read_csv(csv_file, parse_dates=["date_posted"])
            for _, row in hist.iterrows():
                key = (
                    normalize_text(str(row.get("company", ""))) + "|" +
                    normalize_text(str(row.get("title", ""))) + "|" +
                    normalize_text(str(row.get("state", "")))
                )
                if key == "||":
                    continue
                if key not in historical_dates:
                    historical_dates[key] = set()
                if pd.notna(row.get("date_posted")):
                    historical_dates[key].add(str(pd.Timestamp(row["date_posted"]).date()))
                # Also include any previously tracked repost dates
                prev_reposts = row.get("reposted_date", "")
                if pd.notna(prev_reposts) and prev_reposts:
                    for d in str(prev_reposts).split(", "):
                        d = d.strip()
                        if d:
                            historical_dates[key].add(d)
        except Exception as e:
            logger.warning(f"Could not read historical file {csv_file}: {e}")
            continue

    if not historical_dates:
        return df

    # Merge historical dates into current results
    updated = 0
    for idx, row in df.iterrows():
        key = (
            normalize_text(str(row.get("company", ""))) + "|" +
            normalize_text(str(row.get("title", ""))) + "|" +
            normalize_text(str(row.get("state", "")))
        )
        if key == "||" or key not in historical_dates:
            continue

        hist_dates = historical_dates[key]
        original = str(pd.Timestamp(row["date_posted"]).date()) if pd.notna(row["date_posted"]) else None

        # Combine current repost dates with historical ones
        current_reposts = set()
        if row.get("reposted_date"):
            for d in str(row["reposted_date"]).split(", "):
                d = d.strip()
                if d:
                    current_reposts.add(d)

        all_dates = hist_dates | current_reposts
        if original:
            all_dates.add(original)

        repost_dates = sorted(d for d in all_dates if d != original)
        new_repost_str = ", ".join(repost_dates)

        if new_repost_str != (row.get("reposted_date") or ""):
            df.at[idx, "reposted_date"] = new_repost_str
            updated += 1

    if updated > 0:
        logger.info(f"Historical repost check: updated {updated} jobs with repost dates from previous runs")

    return df
```

---

## File: src/exporter.py

```python
"""Export job results to CSV and Excel. Uses a single master file with merge-on-save."""

import logging
from pathlib import Path

import pandas as pd

from .config import OutputConfig, AppConfig, PROJECT_ROOT
from .dedup import deduplicate

logger = logging.getLogger(__name__)


def get_master_path(config: OutputConfig, extension: str = "csv") -> Path:
    """Return path to the single master output file (e.g. data/pharma_jobs.csv)."""
    output_dir = PROJECT_ROOT / config.directory
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{config.filename_prefix}.{extension}"


def migrate_old_files(config: OutputConfig) -> pd.DataFrame | None:
    """One-time migration: merge all old timestamped CSV files into a single DataFrame.
    Returns the merged DataFrame, or None if no old files exist."""
    output_dir = PROJECT_ROOT / config.directory
    if not output_dir.exists():
        return None

    pattern = f"{config.filename_prefix}_*.csv"
    old_files = sorted(output_dir.glob(pattern))
    if not old_files:
        return None

    logger.info(f"Migrating {len(old_files)} old timestamped file(s) into master CSV...")
    dfs = []
    for f in old_files:
        try:
            df = pd.read_csv(f, parse_dates=["date_posted"])
            dfs.append(df)
            logger.info(f"  Read {len(df)} rows from {f.name}")
        except Exception as e:
            logger.warning(f"  Could not read {f.name}: {e}")

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)
    # Deduplicate the historical data
    combined = deduplicate(combined)
    logger.info(f"Migration: {sum(len(d) for d in dfs)} total rows -> {len(combined)} after dedup")

    # Remove old timestamped files
    for f in old_files:
        f.unlink()
        logger.info(f"  Removed old file: {f.name}")

    # Also remove old timestamped xlsx files
    xlsx_pattern = f"{config.filename_prefix}_*.xlsx"
    for f in output_dir.glob(xlsx_pattern):
        f.unlink()
        logger.info(f"  Removed old file: {f.name}")

    return combined


def merge_and_export_csv(new_df: pd.DataFrame, config: AppConfig) -> Path:
    """Merge new results with existing master CSV, deduplicate, and save.

    1. Load existing pharma_jobs.csv (if it exists)
    2. Concat with new_df
    3. Deduplicate (handles URL + fuzzy dedup + repost date merging)
    4. Re-apply discipline filter
    5. Sort by date, recalculate days_since_posted
    6. Save back to pharma_jobs.csv
    """
    from .aggregator import apply_discipline_filter

    master_path = get_master_path(config.output, "csv")

    # On first use, migrate old timestamped files
    existing = None
    if not master_path.exists():
        existing = migrate_old_files(config.output)

    # Load existing master if present
    if master_path.exists():
        try:
            existing = pd.read_csv(master_path, parse_dates=["date_posted"])
            logger.info(f"Loaded existing master CSV: {len(existing)} rows")
        except Exception as e:
            logger.warning(f"Could not read existing master CSV: {e}")
            existing = None

    # Combine existing + new
    if existing is not None and not existing.empty:
        combined = pd.concat([existing.dropna(axis=1, how="all"), new_df.dropna(axis=1, how="all")], ignore_index=True)
        logger.info(f"Combined existing ({len(existing)}) + new ({len(new_df)}) = {len(combined)} rows")
    else:
        combined = new_df.copy()

    # Deduplicate the combined data
    combined = deduplicate(combined)

    # Re-apply discipline filter
    combined = apply_discipline_filter(combined, config.search)

    # Sort by date descending
    if "date_posted" in combined.columns:
        combined["date_posted"] = pd.to_datetime(combined["date_posted"], errors="coerce")
        combined = combined.sort_values("date_posted", ascending=False, na_position="last")
        combined = combined.reset_index(drop=True)

    # Recalculate days_since_posted
    if "date_posted" in combined.columns:
        today = pd.Timestamp.now().normalize()
        combined["days_since_posted"] = (today - combined["date_posted"]).dt.days

    # Ensure reposted_date column exists
    if "reposted_date" not in combined.columns:
        combined["reposted_date"] = ""

    # Save master CSV
    combined.to_csv(master_path, index=False)
    logger.info(f"Master CSV saved: {master_path} ({len(combined)} rows)")

    # Also save Excel
    _export_excel(combined, config.output)

    return master_path


def _export_excel(df: pd.DataFrame, config: OutputConfig) -> Path:
    """Export DataFrame to a single master Excel file with auto-adjusted column widths."""
    path = get_master_path(config, "xlsx")

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Jobs")
        worksheet = writer.sheets["Jobs"]

        # Auto-adjust column widths
        for col_idx, col_name in enumerate(df.columns, 1):
            max_length = len(str(col_name))
            for row_val in df[col_name].head(100):
                cell_len = len(str(row_val)) if pd.notna(row_val) else 0
                max_length = max(max_length, min(cell_len, 60))
            worksheet.column_dimensions[worksheet.cell(1, col_idx).column_letter].width = max_length + 2

    logger.info(f"Excel saved: {path} ({len(df)} rows)")
    return path
```

---

## File: src/dashboard.py

```python
"""Streamlit web dashboard for browsing job search results and evaluation results."""

import json
import os
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# When Streamlit runs this file directly, the parent package isn't on sys.path
_src_dir = Path(__file__).resolve().parent
_project_root = _src_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

from src.config import build_config, PROJECT_ROOT
from src.exporter import get_master_path

st.set_page_config(
    page_title="Pharma/Biotech Job Search",
    page_icon="\U0001f52c",
    layout="wide",
)

REVIEWED_PATH = PROJECT_ROOT / "data" / "reviewed.json"


def load_reviewed() -> dict:
    """Load reviewed timestamps from JSON file. Returns {job_url: timestamp_str}."""
    if REVIEWED_PATH.exists():
        try:
            with open(REVIEWED_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_reviewed(reviewed: dict):
    """Save reviewed timestamps to JSON file."""
    REVIEWED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REVIEWED_PATH, "w") as f:
        json.dump(reviewed, f, indent=2)


def mark_reviewed(job_url: str):
    """Mark a job as reviewed with current timestamp."""
    reviewed = load_reviewed()
    reviewed[job_url] = datetime.now().strftime("%Y-%m-%d %H:%M")
    save_reviewed(reviewed)


def mark_unreviewed(job_url: str):
    """Remove reviewed status from a job."""
    reviewed = load_reviewed()
    reviewed.pop(job_url, None)
    save_reviewed(reviewed)


def extract_job_code(url: str, source: str) -> str:
    """Extract a job code/ID from the job URL based on the source board."""
    if not url or not isinstance(url, str):
        return ""
    try:
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")

        if "linkedin.com" in (parsed.netloc or "") or source == "linkedin":
            match = re.search(r"/jobs/view/(\d+)", path)
            if match:
                return f"LI-{match.group(1)}"

        if "indeed.com" in (parsed.netloc or "") or source == "indeed":
            params = parse_qs(parsed.query)
            if "jk" in params:
                return f"IN-{params['jk'][0]}"
            if "rx_jobId" in params:
                return f"IN-{params['rx_jobId'][0]}"
            segments = path.split("/")
            if segments:
                last = segments[-1]
                match = re.search(r"([a-f0-9]{16,})$", last)
                if match:
                    return f"IN-{match.group(1)[:12]}"

        if "ziprecruiter.com" in (parsed.netloc or "") or source == "zip_recruiter":
            match = re.search(r"/job/([^/?]+)", path) or re.search(r"/jobs/([^/?]+)", path)
            if match:
                return f"ZR-{match.group(1)[:20]}"

        if "glassdoor.com" in (parsed.netloc or "") or source == "glassdoor":
            params = parse_qs(parsed.query)
            if "jobListingId" in params:
                return f"GD-{params['jobListingId'][0]}"

        if source == "google":
            params = parse_qs(parsed.query)
            if "htidocid" in params:
                return f"GJ-{params['htidocid'][0][:12]}"

        if source == "adzuna":
            match = re.search(r"/(\d+)\b", path)
            if match:
                return f"AZ-{match.group(1)}"

        if source == "jooble":
            match = re.search(r"/(\d+)", path)
            if match:
                return f"JB-{match.group(1)}"

        if path:
            last_seg = path.split("/")[-1]
            if last_seg:
                return last_seg[:20]

    except Exception:
        pass
    return ""


def load_data() -> pd.DataFrame:
    """Load data from the master CSV file."""
    config = build_config()
    master_path = get_master_path(config.output, "csv")

    if not master_path.exists():
        # Fallback: try old timestamped files
        data_dir = PROJECT_ROOT / config.output.directory
        if data_dir.exists():
            csv_files = sorted(data_dir.glob(f"{config.output.filename_prefix}_*.csv"), reverse=True)
            if csv_files:
                df = pd.read_csv(csv_files[0], parse_dates=["date_posted"])
                mtime = datetime.fromtimestamp(os.path.getmtime(csv_files[0]))
                st.sidebar.info(f"Last search: {mtime.strftime('%b %d, %Y at %I:%M %p')}")
                return df
        return pd.DataFrame()

    df = pd.read_csv(master_path, parse_dates=["date_posted"])
    mtime = datetime.fromtimestamp(os.path.getmtime(master_path))
    st.sidebar.success(f"Last search: {mtime.strftime('%b %d, %Y at %I:%M %p')} ({len(df)} jobs)")
    return df


def run_search_from_dashboard():
    """Run a new search from the dashboard with progress tracking."""
    from src.aggregator import run_search

    config = build_config()

    status = st.status("Searching all job boards in parallel...", expanded=True)
    progress_bar = status.progress(0.0, text="Starting 5 scrapers in parallel...")
    status.write("Scrapers launched: Indeed, LinkedIn, USAJobs, Adzuna, Jooble")
    status.write("Each scraper saves results to CSV as soon as it finishes.")

    completed_count = 0
    total_scrapers = 5
    total_results = 0

    def progress_callback(scraper_name, scraper_status, count=0):
        nonlocal completed_count, total_results
        if scraper_status == "done":
            completed_count += 1
            total_results += count
            pct = completed_count / total_scrapers
            remaining = total_scrapers - completed_count
            status.write(f"  {scraper_name} done — {count} results (saved to CSV)")
            progress_bar.progress(pct, text=f"{completed_count}/{total_scrapers} scrapers done ({remaining} remaining)...")

    # Run the search — each scraper saves to CSV progressively
    df = run_search(config, progress_callback=progress_callback)

    if df.empty:
        status.update(label="Search complete — no results found.", state="error")
        return

    status.update(
        label=f"Search complete! {len(df)} total jobs in master CSV.",
        state="complete",
    )


# ---------------------------------------------------------------------------
# Shared AG Grid helpers
# ---------------------------------------------------------------------------

DATE_COMPARATOR = JsCode("""
    function(filterLocalDateAtMidnight, cellValue) {
        if (!cellValue) return -1;
        var parts = cellValue.split('-');
        var cellDate = new Date(Number(parts[0]), Number(parts[1]) - 1, Number(parts[2]));
        if (filterLocalDateAtMidnight.getTime() === cellDate.getTime()) return 0;
        return cellDate < filterLocalDateAtMidnight ? -1 : 1;
    }
""")

LINK_RENDERER = JsCode("""
    class LinkRenderer {
        init(params) {
            this.eGui = document.createElement('a');
            this.eGui.innerText = 'View Job';
            this.eGui.setAttribute('href', params.value);
            this.eGui.setAttribute('target', '_blank');
            this.eGui.style.color = '#1a73e8';
            this.eGui.style.textDecoration = 'underline';
        }
        getGui() { return this.eGui; }
    }
""")

DIRECT_LINK_RENDERER = JsCode("""
    class DirectLinkRenderer {
        init(params) {
            this.eGui = document.createElement('span');
            if (params.value && params.value !== 'nan' && params.value !== '') {
                var a = document.createElement('a');
                a.innerText = 'Apply Direct';
                a.setAttribute('href', params.value);
                a.setAttribute('target', '_blank');
                a.style.color = '#e67c00';
                a.style.textDecoration = 'underline';
                this.eGui.appendChild(a);
            }
        }
        getGui() { return this.eGui; }
    }
""")

FIT_SCORE_STYLE = JsCode("""
    function(params) {
        if (params.value >= 70) return {backgroundColor: '#d4edda', fontWeight: 'bold'};
        if (params.value >= 55) return {backgroundColor: '#fff3cd'};
        if (params.value >= 40) return {backgroundColor: '#ffeaa7'};
        return {backgroundColor: '#f8d7da'};
    }
""")


# ---------------------------------------------------------------------------
# Tab 1: Job Listings
# ---------------------------------------------------------------------------

def render_job_listings_tab(df: pd.DataFrame, reviewed_data: dict):
    """Render the Job Listings tab (original dashboard)."""

    # Track grid version to force fresh render after review actions
    if "grid_version" not in st.session_state:
        st.session_state.grid_version = 0

    # --- Sidebar Filters ---
    st.sidebar.header("Job Listings Filters")

    reviewed_count = (df["reviewed_at"] != "").sum()
    unreviewed_count = len(df) - reviewed_count
    show_unreviewed_only = st.sidebar.checkbox(
        f"Unreviewed only ({unreviewed_count} remaining)", value=True, key="jl_unreviewed"
    )
    if show_unreviewed_only:
        df = df[df["reviewed_at"] == ""]

    if "source" in df.columns:
        sources = sorted(df["source"].dropna().unique())
        selected_sources = st.sidebar.multiselect("Job Board", sources, default=sources, key="jl_sources")
        if selected_sources:
            df = df[df["source"].isin(selected_sources)]

    if "state" in df.columns:
        states = sorted(df["state"].dropna().unique())
        states = [s for s in states if s]
        if states:
            selected_states = st.sidebar.multiselect("State", states, key="jl_states")
            if selected_states:
                df = df[df["state"].isin(selected_states)]

    if "is_remote" in df.columns:
        remote_option = st.sidebar.radio("Remote", ["All", "Remote Only", "On-site Only"], key="jl_remote")
        if remote_option == "Remote Only":
            df = df[df["is_remote"] == True]
        elif remote_option == "On-site Only":
            df = df[df["is_remote"] != True]

    if "salary_min" in df.columns:
        has_salary = df["salary_min"].notna().any()
        if has_salary:
            show_salary_only = st.sidebar.checkbox("Only show jobs with salary info", key="jl_salary")
            if show_salary_only:
                df = df[df["salary_min"].notna() | df["salary_max"].notna()]

    st.sidebar.markdown("---")
    show_reposts_only = st.sidebar.checkbox("Only show reposted jobs", key="jl_reposts")
    if show_reposts_only:
        df = df[df["reposted_date"] != ""]

    # --- Main Content ---
    st.markdown(f"**{len(df)} jobs found**")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Jobs", len(df))
    with col2:
        if "source" in df.columns:
            st.metric("Sources", df["source"].nunique())
    with col3:
        if "state" in df.columns:
            st.metric("States", df["state"].replace("", pd.NA).dropna().nunique())
    with col4:
        repost_count = (df["reposted_date"] != "").sum()
        st.metric("Reposted", repost_count)
    with col5:
        st.metric("Reviewed", reviewed_count)

    if "source" in df.columns:
        with st.expander("Source Breakdown"):
            source_counts = df["source"].value_counts()
            st.bar_chart(source_counts)

    if "state" in df.columns:
        with st.expander("Top States"):
            state_counts = df["state"].replace("", pd.NA).dropna().value_counts().head(15)
            st.bar_chart(state_counts)

    # --- AG Grid Table ---
    grid_header_left, grid_header_right = st.columns([3, 1])
    with grid_header_left:
        st.subheader("Job Listings")
    review_button_placeholder = grid_header_right.empty()
    st.caption("Select a row, then click 'Mark as Reviewed' at the top right.")

    display_cols = ["job_code", "title", "company", "location", "state",
                    "date_posted", "reposted_date", "days_since_posted",
                    "source", "job_url", "job_url_direct", "salary_min", "salary_max", "is_remote", "job_type",
                    "reviewed_at"]
    display_cols = [c for c in display_cols if c in df.columns]
    grid_df = df[display_cols].copy()

    gb = GridOptionsBuilder.from_dataframe(grid_df)
    gb.configure_default_column(filterable=True, sortable=True, resizable=True, filter=True)

    for col in ["job_code", "title", "company", "location", "state", "source", "job_type",
                "reposted_date", "reviewed_at"]:
        if col in display_cols:
            gb.configure_column(col, filter="agTextColumnFilter")

    if "date_posted" in display_cols:
        gb.configure_column("date_posted", filter="agDateColumnFilter",
                            filterParams={"comparator": DATE_COMPARATOR})

    for col in ["salary_min", "salary_max", "days_since_posted"]:
        if col in display_cols:
            gb.configure_column(col, filter="agNumberColumnFilter")

    if "job_url" in display_cols:
        gb.configure_column("job_url", headerName="Job Link", cellRenderer=LINK_RENDERER,
                            filter="agTextColumnFilter")

    if "job_url_direct" in display_cols:
        gb.configure_column("job_url_direct", headerName="Direct Link",
                            cellRenderer=DIRECT_LINK_RENDERER, filter="agTextColumnFilter")

    if "is_remote" in display_cols:
        gb.configure_column("is_remote", filter="agSetColumnFilter")

    header_map = {
        "job_code": "Job Code", "title": "Title", "company": "Company",
        "location": "Location", "state": "State", "date_posted": "Posted",
        "reposted_date": "Reposted Date(s)", "days_since_posted": "Days Old",
        "source": "Source", "salary_min": "Salary Min", "salary_max": "Salary Max",
        "is_remote": "Remote", "job_type": "Job Type", "reviewed_at": "Reviewed At",
    }
    for col, name in header_map.items():
        if col in display_cols:
            gb.configure_column(col, headerName=name)

    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=50)
    gb.configure_selection("multiple", use_checkbox=True)
    grid_options = gb.build()
    grid_options["defaultColDef"]["floatingFilter"] = True
    grid_options["defaultColDef"]["suppressSizeToFit"] = True
    grid_options["suppressColumnVirtualisation"] = True

    grid_response = AgGrid(
        grid_df, gridOptions=grid_options, update_on=["selectionChanged"],
        allow_unsafe_jscode=True, theme="streamlit", height=600,
        key=f"job_grid_{st.session_state.grid_version}",
    )

    selected = grid_response.get("selected_rows", None)
    selected_rows = []
    if selected is not None:
        if hasattr(selected, "iterrows"):
            selected_rows = [row for _, row in selected.iterrows()]
        elif isinstance(selected, list):
            selected_rows = selected

    if selected_rows:
        urls = [r.get("job_url", "") for r in selected_rows if r.get("job_url", "")]
        all_reviewed = all(r.get("reviewed_at", "") for r in selected_rows)
        with review_button_placeholder.container():
            if all_reviewed:
                if st.button(f"Undo Review ({len(urls)})", key="jl_unreview_selected"):
                    for u in urls:
                        mark_unreviewed(u)
                    st.session_state.grid_version += 1
                    st.rerun()
            else:
                if st.button(f"Mark as Reviewed ({len(urls)})", key="jl_review_selected", type="primary"):
                    for u in urls:
                        mark_reviewed(u)
                    st.session_state.grid_version += 1
                    st.rerun()

    if len(selected_rows) == 1:
        row = selected_rows[0]
        st.subheader("Selected Job Details")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Title:** {row.get('title', 'N/A')}")
            st.markdown(f"**Company:** {row.get('company', 'N/A')}")
            st.markdown(f"**Location:** {row.get('location', 'N/A')}")
            st.markdown(f"**Source:** {row.get('source', 'N/A')}")
        with col_b:
            st.markdown(f"**Job Code:** {row.get('job_code', 'N/A')}")
            st.markdown(f"**Posted:** {row.get('date_posted', 'N/A')}")
            days = row.get("days_since_posted")
            if pd.notna(days):
                st.markdown(f"**Days Old:** {int(days)}")
            repost = row.get("reposted_date", "")
            if repost:
                st.markdown(f"**Reposted on:** {repost}")
            reviewed_ts = row.get("reviewed_at", "")
            if reviewed_ts:
                st.markdown(f"**Reviewed:** {reviewed_ts}")
        url = row.get("job_url", "")
        direct_url = row.get("job_url_direct", "")
        link_parts = []
        if url:
            link_parts.append(f"[View on Job Board]({url})")
        if direct_url and str(direct_url) not in ("", "nan"):
            link_parts.append(f"[Apply Direct (Employer Site)]({direct_url})")
        if link_parts:
            st.markdown(" | ".join(link_parts))
    elif len(selected_rows) > 1:
        st.info(f"{len(selected_rows)} jobs selected — click 'Mark as Reviewed' above to review them all.")

    # Expandable job descriptions
    st.subheader("Job Descriptions")
    if "description" in df.columns:
        for idx, row in df.head(50).iterrows():
            title = row.get("title", "Untitled")
            company = row.get("company", "Unknown")
            job_code = row.get("job_code", "")
            label = f"[{job_code}] {title} - {company}" if job_code else f"{title} - {company}"
            desc = row.get("description", "")
            if pd.notna(desc) and desc:
                with st.expander(label):
                    st.markdown(f"**Location:** {row.get('location', 'N/A')}")
                    st.markdown(f"**Source:** {row.get('source', 'N/A')}")
                    days = row.get("days_since_posted")
                    if pd.notna(days):
                        st.markdown(f"**Days old:** {int(days)}")
                    repost = row.get("reposted_date", "")
                    if repost:
                        st.markdown(f"**Reposted on:** {repost}")
                    link_parts = []
                    if pd.notna(row.get("job_url")):
                        link_parts.append(f"[View on Job Board]({row['job_url']})")
                    direct = row.get("job_url_direct", "")
                    if pd.notna(direct) and str(direct) not in ("", "nan"):
                        link_parts.append(f"[Apply Direct]({direct})")
                    if link_parts:
                        st.markdown(" | ".join(link_parts))
                    st.markdown("---")
                    st.markdown(str(desc)[:3000])


# ---------------------------------------------------------------------------
# Tab 2: Evaluation Results
# ---------------------------------------------------------------------------

def load_evaluation_data(df: pd.DataFrame) -> pd.DataFrame:
    """Load evaluation results and merge with job data."""
    config = build_config()
    eval_path = PROJECT_ROOT / config.evaluation.evaluations_store

    if not eval_path.exists():
        return pd.DataFrame()

    try:
        with open(eval_path, "r") as f:
            eval_data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return pd.DataFrame()

    evals = eval_data.get("evaluations", {})
    if not evals:
        return pd.DataFrame()

    eval_records = []
    for url, ev in evals.items():
        record = {"job_url": url}
        record.update(ev)
        if isinstance(record.get("matching_skills"), list):
            record["matching_skills"] = ", ".join(record["matching_skills"])
        if isinstance(record.get("missing_skills"), list):
            record["missing_skills"] = ", ".join(record["missing_skills"])
        eval_records.append(record)

    eval_df = pd.DataFrame(eval_records)

    if df.empty or eval_df.empty:
        return eval_df

    merged = df.merge(eval_df, on="job_url", how="inner")
    return merged


def render_evaluation_tab(df: pd.DataFrame, reviewed_data: dict):
    """Render the Evaluation Results tab."""

    if "eval_grid_version" not in st.session_state:
        st.session_state.eval_grid_version = 0

    eval_df = load_evaluation_data(df)

    if eval_df.empty:
        st.warning("No evaluation results found. Run evaluation first:\n\n"
                    "`python job_search.py --evaluate-only --eval-days 1`")
        return

    # Merge reviewed timestamps
    eval_df["reviewed_at"] = eval_df["job_url"].map(reviewed_data).fillna("")

    # --- Sidebar Filters ---
    st.sidebar.header("Evaluation Filters")

    min_score = st.sidebar.slider("Min Fit Score", 0, 100, 0, key="eval_min_score")
    if min_score > 0:
        eval_df = eval_df[eval_df["fit_score"] >= min_score]

    if "recommendation" in eval_df.columns:
        recs = sorted(eval_df["recommendation"].dropna().unique())
        selected_recs = st.sidebar.multiselect("Recommendation", recs, default=recs, key="eval_recs")
        if selected_recs:
            eval_df = eval_df[eval_df["recommendation"].isin(selected_recs)]

    if "fit_bucket" in eval_df.columns:
        buckets = sorted(eval_df["fit_bucket"].dropna().unique())
        selected_buckets = st.sidebar.multiselect("Fit Bucket", buckets, default=buckets, key="eval_buckets")
        if selected_buckets:
            eval_df = eval_df[eval_df["fit_bucket"].isin(selected_buckets)]

    reviewed_count = (eval_df["reviewed_at"] != "").sum()
    unreviewed_count = len(eval_df) - reviewed_count
    show_unreviewed = st.sidebar.checkbox(
        f"Unreviewed only ({unreviewed_count} remaining)", value=True, key="eval_unreviewed"
    )
    if show_unreviewed:
        eval_df = eval_df[eval_df["reviewed_at"] == ""]

    if "source" in eval_df.columns:
        sources = sorted(eval_df["source"].dropna().unique())
        selected_sources = st.sidebar.multiselect("Source", sources, default=sources, key="eval_sources")
        if selected_sources:
            eval_df = eval_df[eval_df["source"].isin(selected_sources)]

    if "state" in eval_df.columns:
        states = sorted(eval_df["state"].dropna().unique())
        states = [s for s in states if s]
        if states:
            selected_states = st.sidebar.multiselect("State", states, key="eval_states")
            if selected_states:
                eval_df = eval_df[eval_df["state"].isin(selected_states)]

    if "is_remote" in eval_df.columns:
        remote_option = st.sidebar.radio("Remote", ["All", "Remote Only", "On-site Only"], key="eval_remote")
        if remote_option == "Remote Only":
            eval_df = eval_df[eval_df["is_remote"] == True]
        elif remote_option == "On-site Only":
            eval_df = eval_df[eval_df["is_remote"] != True]

    # --- Summary Metrics ---
    total_eval = len(eval_df)
    apply_count = len(eval_df[eval_df.get("recommendation", pd.Series()) == "apply"]) if "recommendation" in eval_df.columns else 0
    maybe_count = len(eval_df[eval_df.get("recommendation", pd.Series()) == "maybe"]) if "recommendation" in eval_df.columns else 0
    skip_count = len(eval_df[eval_df.get("recommendation", pd.Series()) == "skip"]) if "recommendation" in eval_df.columns else 0
    avg_score = round(eval_df["fit_score"].mean(), 1) if "fit_score" in eval_df.columns and total_eval > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Evaluated", total_eval)
    with c2:
        st.metric("Apply", apply_count)
    with c3:
        st.metric("Maybe", maybe_count)
    with c4:
        st.metric("Skip", skip_count)
    with c5:
        st.metric("Avg Score", avg_score)

    # --- AG Grid ---
    grid_header_left, grid_header_right = st.columns([3, 1])
    with grid_header_left:
        st.subheader("Evaluation Results")
    eval_review_placeholder = grid_header_right.empty()

    # Prepare display columns
    display_cols = [
        "fit_score", "fit_bucket", "recommendation",
        "title", "company", "domain_match",
        "location", "state", "date_posted", "days_since_posted",
        "source", "job_url", "job_url_direct",
        "matching_skills", "missing_skills",
        "evaluated_timestamp", "reviewed_at",
    ]
    display_cols = [c for c in display_cols if c in eval_df.columns]

    # Compute days_since_posted if not present
    if "days_since_posted" not in eval_df.columns and "date_posted" in eval_df.columns:
        eval_df["date_posted"] = pd.to_datetime(eval_df["date_posted"], errors="coerce")
        today = pd.Timestamp.now().normalize()
        eval_df["days_since_posted"] = (today - eval_df["date_posted"]).dt.days
        eval_df["date_posted"] = eval_df["date_posted"].dt.strftime("%Y-%m-%d")
        if "days_since_posted" not in display_cols:
            display_cols.insert(display_cols.index("date_posted") + 1, "days_since_posted")

    grid_df = eval_df[display_cols].copy()

    # Sort by fit_score descending
    if "fit_score" in grid_df.columns:
        grid_df = grid_df.sort_values("fit_score", ascending=False).reset_index(drop=True)

    gb = GridOptionsBuilder.from_dataframe(grid_df)
    gb.configure_default_column(filterable=True, sortable=True, resizable=True, filter=True)

    # Fit score with color coding
    if "fit_score" in display_cols:
        gb.configure_column("fit_score", headerName="Fit Score",
                            filter="agNumberColumnFilter", cellStyle=FIT_SCORE_STYLE)

    if "fit_bucket" in display_cols:
        gb.configure_column("fit_bucket", headerName="Fit Bucket", filter="agSetColumnFilter")

    if "recommendation" in display_cols:
        gb.configure_column("recommendation", headerName="Recommendation", filter="agSetColumnFilter")

    for col in ["title", "company", "domain_match", "location", "state",
                "matching_skills", "missing_skills", "evaluated_timestamp", "reviewed_at"]:
        if col in display_cols:
            gb.configure_column(col, filter="agTextColumnFilter")

    if "date_posted" in display_cols:
        gb.configure_column("date_posted", headerName="Posted",
                            filter="agDateColumnFilter",
                            filterParams={"comparator": DATE_COMPARATOR})

    for col in ["days_since_posted"]:
        if col in display_cols:
            gb.configure_column(col, headerName="Days Old", filter="agNumberColumnFilter")

    if "source" in display_cols:
        gb.configure_column("source", headerName="Source", filter="agSetColumnFilter")

    if "job_url" in display_cols:
        gb.configure_column("job_url", headerName="Job Link",
                            cellRenderer=LINK_RENDERER, filter="agTextColumnFilter")

    if "job_url_direct" in display_cols:
        gb.configure_column("job_url_direct", headerName="Direct Link",
                            cellRenderer=DIRECT_LINK_RENDERER, filter="agTextColumnFilter")

    # Human-friendly header names
    header_map = {
        "title": "Title", "company": "Company", "domain_match": "Domain Match",
        "location": "Location", "state": "State",
        "matching_skills": "Matching Skills", "missing_skills": "Missing Skills",
        "evaluated_timestamp": "Evaluated At", "reviewed_at": "Reviewed At",
    }
    for col, name in header_map.items():
        if col in display_cols:
            gb.configure_column(col, headerName=name)

    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=50)
    gb.configure_selection("multiple", use_checkbox=True)
    grid_options = gb.build()
    grid_options["defaultColDef"]["floatingFilter"] = True
    grid_options["defaultColDef"]["suppressSizeToFit"] = True
    grid_options["suppressColumnVirtualisation"] = True

    grid_response = AgGrid(
        grid_df, gridOptions=grid_options, update_on=["selectionChanged"],
        allow_unsafe_jscode=True, theme="streamlit", height=600,
        key=f"eval_grid_{st.session_state.eval_grid_version}",
    )

    selected = grid_response.get("selected_rows", None)
    selected_rows = []
    if selected is not None:
        if hasattr(selected, "iterrows"):
            selected_rows = [row for _, row in selected.iterrows()]
        elif isinstance(selected, list):
            selected_rows = selected

    # Review buttons
    if selected_rows:
        urls = [r.get("job_url", "") for r in selected_rows if r.get("job_url", "")]
        all_reviewed = all(r.get("reviewed_at", "") for r in selected_rows)
        with eval_review_placeholder.container():
            if all_reviewed:
                if st.button(f"Undo Review ({len(urls)})", key="eval_unreview"):
                    for u in urls:
                        mark_unreviewed(u)
                    st.session_state.eval_grid_version += 1
                    st.rerun()
            else:
                if st.button(f"Mark as Reviewed ({len(urls)})", key="eval_review", type="primary"):
                    for u in urls:
                        mark_reviewed(u)
                    st.session_state.eval_grid_version += 1
                    st.rerun()

    # Detail panel for single selected row
    if len(selected_rows) == 1:
        row = selected_rows[0]
        st.subheader("Evaluation Details")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Title:** {row.get('title', 'N/A')}")
            st.markdown(f"**Company:** {row.get('company', 'N/A')}")
            st.markdown(f"**Location:** {row.get('location', 'N/A')}")
            score = row.get('fit_score', 0)
            bucket = row.get('fit_bucket', 'N/A')
            rec = row.get('recommendation', 'N/A')
            st.markdown(f"**Fit Score:** {score} ({bucket}) — **{rec.upper()}**")
        with col_b:
            st.markdown(f"**Domain Match:** {row.get('domain_match', 'N/A')}")
            st.markdown(f"**Source:** {row.get('source', 'N/A')}")
            st.markdown(f"**Posted:** {row.get('date_posted', 'N/A')}")
            reviewed_ts = row.get("reviewed_at", "")
            if reviewed_ts:
                st.markdown(f"**Reviewed:** {reviewed_ts}")

        # Reasoning
        reasoning = row.get("reasoning", "")
        if reasoning:
            st.markdown(f"**Reasoning:** {reasoning}")

        # Skills as colored tags
        matching = row.get("matching_skills", "")
        missing = row.get("missing_skills", "")
        if matching:
            skills_html = " ".join(
                f'<span style="background-color:#d4edda;padding:2px 8px;border-radius:12px;margin:2px;display:inline-block;font-size:0.85em;">{s.strip()}</span>'
                for s in str(matching).split(",") if s.strip()
            )
            st.markdown(f"**Matching Skills:** {skills_html}", unsafe_allow_html=True)
        if missing:
            skills_html = " ".join(
                f'<span style="background-color:#f8d7da;padding:2px 8px;border-radius:12px;margin:2px;display:inline-block;font-size:0.85em;">{s.strip()}</span>'
                for s in str(missing).split(",") if s.strip()
            )
            st.markdown(f"**Missing Skills:** {skills_html}", unsafe_allow_html=True)

        # Links
        url = row.get("job_url", "")
        direct_url = row.get("job_url_direct", "")
        link_parts = []
        if url:
            link_parts.append(f"[View on Job Board]({url})")
        if direct_url and str(direct_url) not in ("", "nan"):
            link_parts.append(f"[Apply Direct (Employer Site)]({direct_url})")
        if link_parts:
            st.markdown(" | ".join(link_parts))

        # Job description if available
        if "description" in eval_df.columns:
            job_row = eval_df[eval_df["job_url"] == url]
            if not job_row.empty:
                desc = job_row.iloc[0].get("description", "")
                if pd.notna(desc) and str(desc).strip() and str(desc).lower() != "nan":
                    with st.expander("Full Job Description"):
                        st.markdown(str(desc)[:5000])

    elif len(selected_rows) > 1:
        st.info(f"{len(selected_rows)} jobs selected — click 'Mark as Reviewed' above to review them all.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("Pharma/Biotech Job Search Results")

    # --- Sidebar: Search & Reload ---
    st.sidebar.header("Actions")

    if st.sidebar.button("\U0001f50d Run New Search", use_container_width=True, type="primary"):
        run_search_from_dashboard()
        st.cache_data.clear()
        st.rerun()

    if st.sidebar.button("\U0001f504 Reload Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")

    df = load_data()

    if df.empty:
        st.warning("No data found. Click **Run New Search** or run from CLI: `python job_search.py`")
        return

    # Compute derived columns
    if "date_posted" in df.columns:
        df["date_posted"] = pd.to_datetime(df["date_posted"], errors="coerce")
        today = pd.Timestamp.now().normalize()
        df["days_since_posted"] = (today - df["date_posted"]).dt.days
        df["date_posted"] = df["date_posted"].dt.strftime("%Y-%m-%d")

    if "reposted_date" not in df.columns:
        df["reposted_date"] = ""
    df["reposted_date"] = df["reposted_date"].fillna("")

    df["job_code"] = df.apply(
        lambda row: extract_job_code(row.get("job_url", ""), row.get("source", "")),
        axis=1,
    )

    # Load reviewed data (shared across both tabs)
    reviewed_data = load_reviewed()
    df["reviewed_at"] = df["job_url"].map(reviewed_data).fillna("")

    # --- Tabs ---
    tab1, tab2 = st.tabs(["Job Listings", "Evaluation Results"])

    with tab1:
        render_job_listings_tab(df.copy(), reviewed_data)

    with tab2:
        render_evaluation_tab(df.copy(), reviewed_data)


if __name__ == "__main__":
    main()
```

---

## File: src/scraper_jobspy.py

```python
"""JobSpy wrapper: iterates search terms, collects results into a DataFrame."""

import logging
import time

import pandas as pd

from .config import SearchConfig

logger = logging.getLogger(__name__)


def scrape_single_term(config: SearchConfig, site: str, term: str) -> pd.DataFrame:
    """Scrape a single (site, term) pair. Called by work queue workers.

    Uses priority_results_per_site for terms in priority_terms list,
    otherwise uses results_per_site.
    """
    try:
        from jobspy import scrape_jobs as jobspy_scrape
    except ImportError:
        logger.error("python-jobspy is not installed. Run: pip install python-jobspy")
        return pd.DataFrame()

    # Check if this is a priority term (case-insensitive)
    priority_lower = [t.lower() for t in config.priority_terms]
    if term.lower() in priority_lower:
        results_wanted = config.priority_results_per_site
    else:
        results_wanted = config.results_per_site

    hours_old = config.days * 24

    logger.info(f"Searching for '{term}' on [{site}] (results_wanted={results_wanted})...")

    try:
        df = jobspy_scrape(
            site_name=[site],
            search_term=term,
            location=config.location,
            country_indeed=config.country_indeed,
            hours_old=hours_old,
            results_wanted=results_wanted,
            description_format="markdown",
            verbose=0,
        )

        if df is not None and not df.empty:
            df["search_term"] = term
            logger.info(f"  Found {len(df)} results for '{term}' on {site}")
            return df
        else:
            logger.info(f"  No results for '{term}' on {site}")
            return pd.DataFrame()

    except Exception as e:
        logger.warning(f"  Error searching for '{term}' on {site}: {e}")
        return pd.DataFrame()


def scrape_jobs(config: SearchConfig, site: str = None) -> pd.DataFrame:
    """Run JobSpy scrape_jobs for each search term, concatenate results.

    Args:
        config: Search configuration.
        site: Optional single site to search (e.g. "indeed" or "linkedin").
              If None, searches all sites in config.sites.
    """
    try:
        from jobspy import scrape_jobs as jobspy_scrape
    except ImportError:
        logger.error("python-jobspy is not installed. Run: pip install python-jobspy")
        return pd.DataFrame()

    sites = [site] if site else config.sites
    all_results = []
    hours_old = config.days * 24

    for i, term in enumerate(config.terms):
        if i > 0:
            logger.info(f"Rate limiting: waiting {config.delay_between_searches}s...")
            time.sleep(config.delay_between_searches)

        logger.info(f"Searching for '{term}' on {sites}...")

        try:
            df = jobspy_scrape(
                site_name=sites,
                search_term=term,
                location=config.location,
                country_indeed=config.country_indeed,
                hours_old=hours_old,
                results_wanted=config.results_per_site,
                description_format="markdown",
                verbose=0,
            )

            if df is not None and not df.empty:
                df["search_term"] = term
                all_results.append(df)
                logger.info(f"  Found {len(df)} results for '{term}'")
            else:
                logger.info(f"  No results for '{term}'")

        except Exception as e:
            logger.warning(f"  Error searching for '{term}': {e}")
            continue

    if not all_results:
        logger.warning("No results found across all search terms.")
        return pd.DataFrame()

    combined = pd.concat([r.dropna(axis=1, how="all") for r in all_results], ignore_index=True)
    logger.info(f"Total raw results: {len(combined)}")
    return combined
```

---

## File: src/scraper_usajobs.py

```python
"""USAJobs API client: queries federal pharma/biotech jobs, normalizes to common schema."""

import logging
from datetime import datetime, timedelta

import pandas as pd
import requests

from .config import USAJobsConfig

logger = logging.getLogger(__name__)

USAJOBS_BASE_URL = "https://data.usajobs.gov/api/search"


def scrape_usajobs(config: USAJobsConfig, days: int = 7) -> pd.DataFrame:
    """Query USAJobs API for pharma/biotech roles. Returns empty DataFrame if not configured."""
    if not config.enabled:
        logger.info("USAJobs: skipping (no API key configured)")
        return pd.DataFrame()

    api_key = config.get_api_key()
    email = config.get_email()

    headers = {
        "Authorization-Key": api_key,
        "User-Agent": email,
        "Host": "data.usajobs.gov",
    }

    date_from = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    all_jobs = []

    for keyword in config.keywords:
        logger.info(f"USAJobs: searching for '{keyword}'...")
        try:
            params = {
                "Keyword": keyword,
                "DatePosted": days,
                "ResultsPerPage": 100,
            }
            resp = requests.get(USAJOBS_BASE_URL, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("SearchResult", {}).get("SearchResultItems", [])
            for item in results:
                pos = item.get("MatchedObjectDescriptor", {})
                locations = pos.get("PositionLocation", [])
                location_str = locations[0].get("LocationName", "") if locations else ""

                # Extract state from location
                state = ""
                if ", " in location_str:
                    parts = location_str.split(", ")
                    state = parts[-1].strip() if len(parts) >= 2 else ""

                salary_min = None
                salary_max = None
                remuneration = pos.get("PositionRemuneration", [])
                if remuneration:
                    salary_min = remuneration[0].get("MinimumRange")
                    salary_max = remuneration[0].get("MaximumRange")
                    try:
                        salary_min = float(salary_min) if salary_min else None
                        salary_max = float(salary_max) if salary_max else None
                    except (ValueError, TypeError):
                        salary_min = salary_max = None

                date_posted = pos.get("PublicationStartDate", "")
                try:
                    date_posted = datetime.strptime(date_posted, "%Y-%m-%dT%H:%M:%S.%f").date()
                except (ValueError, TypeError):
                    try:
                        date_posted = datetime.strptime(date_posted, "%Y-%m-%d").date()
                    except (ValueError, TypeError):
                        date_posted = None

                all_jobs.append({
                    "title": pos.get("PositionTitle", ""),
                    "company": pos.get("OrganizationName", ""),
                    "location": location_str,
                    "state": state,
                    "date_posted": date_posted,
                    "source": "usajobs",
                    "job_url": pos.get("PositionURI", ""),
                    "salary_min": salary_min,
                    "salary_max": salary_max,
                    "is_remote": False,
                    "job_type": pos.get("PositionSchedule", [{}])[0].get("Name", "")
                    if pos.get("PositionSchedule") else "",
                    "description": pos.get("QualificationSummary", ""),
                    "search_term": keyword,
                })

            logger.info(f"  USAJobs: found {len(results)} results for '{keyword}'")

        except Exception as e:
            logger.warning(f"  USAJobs error for '{keyword}': {e}")
            continue

    if not all_jobs:
        return pd.DataFrame()

    return pd.DataFrame(all_jobs)
```

---

## File: src/scraper_adzuna.py

```python
"""Adzuna API client: queries pharma/biotech jobs, normalizes to common schema."""

import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from .config import AdzunaConfig

logger = logging.getLogger(__name__)

ADZUNA_BASE_URL = "https://api.adzuna.com/v1/api/jobs/us/search"


def scrape_adzuna(config: AdzunaConfig, days: int = 7) -> pd.DataFrame:
    """Query Adzuna API for pharma/biotech roles. Returns empty DataFrame if not configured."""
    if not config.enabled:
        logger.info("Adzuna: skipping (no API credentials configured)")
        return pd.DataFrame()

    app_id = config.get_app_id()
    app_key = config.get_app_key()
    all_jobs = []

    for keyword in config.keywords:
        logger.info(f"Adzuna: searching for '{keyword}'...")

        for page in range(1, config.max_pages + 1):
            try:
                params = {
                    "app_id": app_id,
                    "app_key": app_key,
                    "what": keyword,
                    "where": "United States",
                    "results_per_page": config.results_per_page,
                    "sort_by": "date",
                    "max_days_old": days,
                }
                url = f"{ADZUNA_BASE_URL}/{page}"
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                results = data.get("results", [])
                if not results:
                    break

                for item in results:
                    location_str = item.get("location", {}).get("display_name", "")

                    salary_min = item.get("salary_min")
                    salary_max = item.get("salary_max")
                    try:
                        salary_min = float(salary_min) if salary_min else None
                        salary_max = float(salary_max) if salary_max else None
                    except (ValueError, TypeError):
                        salary_min = salary_max = None

                    date_posted = item.get("created")
                    try:
                        date_posted = datetime.strptime(
                            date_posted, "%Y-%m-%dT%H:%M:%SZ"
                        ).date()
                    except (ValueError, TypeError):
                        try:
                            date_posted = datetime.strptime(
                                date_posted[:10], "%Y-%m-%d"
                            ).date()
                        except (ValueError, TypeError):
                            date_posted = None

                    is_remote = False
                    if location_str:
                        is_remote = "remote" in location_str.lower()

                    contract_time = item.get("contract_time", "")
                    job_type = ""
                    if contract_time == "full_time":
                        job_type = "Full-time"
                    elif contract_time == "part_time":
                        job_type = "Part-time"
                    elif contract_time:
                        job_type = contract_time

                    all_jobs.append({
                        "title": item.get("title", ""),
                        "company": item.get("company", {}).get("display_name", ""),
                        "location": location_str,
                        "state": "",
                        "date_posted": date_posted,
                        "source": "adzuna",
                        "job_url": item.get("redirect_url", ""),
                        "salary_min": salary_min,
                        "salary_max": salary_max,
                        "is_remote": is_remote,
                        "job_type": job_type,
                        "description": item.get("description", ""),
                    })

                logger.info(
                    f"  Adzuna: page {page} returned {len(results)} results for '{keyword}'"
                )

                if len(results) < config.results_per_page:
                    break

                time.sleep(2.5)

            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    logger.warning(f"  Adzuna rate limited for '{keyword}', stopping pagination")
                    break
                logger.warning(f"  Adzuna error for '{keyword}' page {page}: {e}")
                break
            except Exception as e:
                logger.warning(f"  Adzuna error for '{keyword}' page {page}: {e}")
                break

        logger.info(f"  Adzuna: total {len(all_jobs)} results so far")

    if not all_jobs:
        return pd.DataFrame()

    return pd.DataFrame(all_jobs)
```

---

## File: src/scraper_jooble.py

```python
"""Jooble API client: queries pharma/biotech jobs, normalizes to common schema."""

import logging
import re
import time
from datetime import datetime

import pandas as pd
import requests

from .config import JoobleConfig

logger = logging.getLogger(__name__)

JOOBLE_BASE_URL = "https://jooble.org/api"


def _parse_salary(salary_str: str):
    """Parse Jooble salary string (e.g., '$80,000 - $120,000') into min/max floats."""
    if not salary_str or not isinstance(salary_str, str):
        return None, None
    numbers = re.findall(r"[\d]+\.?\d*", salary_str.replace(",", ""))
    try:
        if len(numbers) >= 2:
            return float(numbers[0]), float(numbers[1])
        elif len(numbers) == 1:
            return float(numbers[0]), None
    except (ValueError, TypeError):
        pass
    return None, None


def scrape_jooble(config: JoobleConfig, days: int = 7) -> pd.DataFrame:
    """Query Jooble API for pharma/biotech roles. Returns empty DataFrame if not configured."""
    if not config.enabled:
        logger.info("Jooble: skipping (no API key configured)")
        return pd.DataFrame()

    api_key = config.get_api_key()
    url = f"{JOOBLE_BASE_URL}/{api_key}"
    all_jobs = []

    for keyword in config.keywords:
        logger.info(f"Jooble: searching for '{keyword}'...")

        for page in range(1, config.max_pages + 1):
            try:
                payload = {
                    "keywords": keyword,
                    "location": "United States",
                    "page": str(page),
                }
                resp = requests.post(url, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                jobs = data.get("jobs", [])
                if not jobs:
                    break

                for item in jobs:
                    location_str = item.get("location", "")

                    salary_min, salary_max = _parse_salary(item.get("salary", ""))

                    date_str = item.get("updated", "")
                    date_posted = None
                    if date_str:
                        try:
                            date_posted = datetime.strptime(
                                date_str[:10], "%Y-%m-%d"
                            ).date()
                        except (ValueError, TypeError):
                            date_posted = None

                    is_remote = False
                    if location_str:
                        is_remote = "remote" in location_str.lower()
                    job_type_str = item.get("type", "")
                    if job_type_str and "remote" in job_type_str.lower():
                        is_remote = True

                    all_jobs.append({
                        "title": item.get("title", ""),
                        "company": item.get("company", ""),
                        "location": location_str,
                        "state": "",
                        "date_posted": date_posted,
                        "source": "jooble",
                        "job_url": item.get("link", ""),
                        "salary_min": salary_min,
                        "salary_max": salary_max,
                        "is_remote": is_remote,
                        "job_type": job_type_str if job_type_str else "",
                        "description": item.get("snippet", ""),
                    })

                logger.info(
                    f"  Jooble: page {page} returned {len(jobs)} results for '{keyword}'"
                )

                total_count = data.get("totalCount", 0)
                if len(all_jobs) >= total_count:
                    break

                time.sleep(2)

            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    logger.warning(
                        f"  Jooble rate limited for '{keyword}', stopping pagination"
                    )
                    break
                logger.warning(f"  Jooble error for '{keyword}' page {page}: {e}")
                break
            except Exception as e:
                logger.warning(f"  Jooble error for '{keyword}' page {page}: {e}")
                break

        logger.info(f"  Jooble: total {len(all_jobs)} results so far")

    if not all_jobs:
        return pd.DataFrame()

    return pd.DataFrame(all_jobs)
```

---

## File: src/evaluator.py

```python
"""Job evaluation pipeline: Stage 1 rule-based pre-filter + Stage 2 Claude API scoring."""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import EvaluationConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1: Rule-based pre-filter
# ---------------------------------------------------------------------------

# Title patterns that auto-skip (case-insensitive regex)
SKIP_TITLE_PATTERNS = [
    # Senior leadership (too senior / wrong track)
    r"\bvp\b", r"\bsvp\b", r"\bceo\b", r"\bcfo\b", r"\bcoo\b", r"\bcmo\b",
    r"\bchief\b.*\bofficer\b",
    # Manufacturing / QC / QA
    r"\bqc analyst\b", r"\bqc technician\b", r"\bqc specialist\b",
    r"\bmanufacturing\b.*\btechnician\b", r"\bmanufacturing\b.*\boperator\b",
    r"\bproduction\b.*\boperator\b", r"\bproduction\b.*\btechnician\b",
    r"\bcgmp\b.*\bmanufactur\b", r"\bfill.*finish\b",
    # Instrumentation-specific mismatches
    r"\bhplc\b.*\bscientist\b", r"\blc-ms\b", r"\blcms\b", r"\bmass spec\b",
    r"\bprotein.*scientist\b", r"\bprotein.*chemist\b",
    r"\bformulation\b.*\bscientist\b",
    # Computational / bioinformatics
    r"\bbioinformatic", r"\bcomputational\b.*\bbiolog",
    r"\bdata\s+scientist\b", r"\bmachine\s+learning\b",
    r"\bsoftware\b.*\bengineer\b", r"\bai\b.*\bengineer\b",
    # Chemistry-specific
    r"\borganic\b.*\bchemist\b", r"\bprocess\b.*\bchemist\b",
    r"\bmedicinal\b.*\bchemist\b", r"\bsynthetic\b.*\bchemist\b",
    r"\banalytical\b.*\bchemist\b",
    # Entry level / trainee
    r"\bresearch\s+associate\b", r"\blab\s+technician\b",
    r"\bresearch\s+technician\b", r"\bintern\b", r"\bco-?op\b",
    r"\bpostdoc\b", r"\bpostdoctoral\b",
    # Clinical / medical
    r"\bmedical\s+director\b", r"\bphysician\b", r"\bnurse\b",
    r"\bpharmacist\b", r"\bclinical\s+research\s+coordinator\b",
    r"\bmedical\s+science\s+liaison\b",
    # Sales / commercial
    r"\bsales\b", r"\bterritory\b.*\bmanager\b", r"\bbusiness\s+development\b",
    r"\bcommercial\b",
    # Other mismatches
    r"\bveterinar\b", r"\bregulatory\s+affairs\b.*\bspecialist\b",
    r"\bproject\s+manager\b", r"\bprogram\s+manager\b",
    r"\bsupply\s+chain\b", r"\bprocurement\b",
    r"\benvironmental\b.*\bscientist\b",
]

# Description patterns that auto-skip (unless rescued)
SKIP_DESCRIPTION_PATTERNS = [
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
RESCUE_PATTERNS = [
    r"\bqpcr\b", r"\brt-qpcr\b", r"\bddpcr\b",
    r"\bflow\s+cytometry\b", r"\bfacs\b",
    r"\bgene\s+therapy\b", r"\bviral\s+vector\b",
    r"\baav\b", r"\blentivir\b", r"\bcar-t\b",
    r"\bglp\b", r"\bbiodistribution\b", r"\bshedding\b",
    r"\blnp\b", r"\bnucleic\s+acid\b",
    r"\borganoid\b", r"\b10x\s+genomics\b",
]

# Boost patterns: jobs matching these get priority for evaluation
BOOST_PATTERNS = [
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


_skip_title_compiled = _compile_patterns(SKIP_TITLE_PATTERNS)
_skip_desc_compiled = _compile_patterns(SKIP_DESCRIPTION_PATTERNS)
_rescue_compiled = _compile_patterns(RESCUE_PATTERNS)
_boost_compiled = _compile_patterns(BOOST_PATTERNS)


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


def _build_system_prompt(profile: dict) -> str:
    """Build the system prompt for Claude API evaluation."""
    return f"""You are a job-fit evaluator for a senior scientist in pharma/biotech.

CANDIDATE PROFILE:
{json.dumps(profile, indent=2)}

DOMAIN CALIBRATION (from historical fit assessments):
- CRO management / bioanalytical outsourcing: 70%+
- Gene therapy bioanalytical (AAV, lentiviral, LNP): 65-75%
- Cell therapy analytical (CAR-T): 65-70%
- Flow cytometry / immunophenotyping: strong fit
- Biomarker / translational: 60-75%
- Cell biology / in vitro assay development: 60-65%
- Molecular diagnostics / qPCR-focused: 60-65%
- NAMs / in vitro tox / 3D models: moderate
- Oncology target validation: 55-60%
- CMC analytical with qPCR/ddPCR: 50-55%
- IVD / analytical validation: 55-60%
- GMP AS&T / QC analytical (with qPCR overlap): 55-60%
- Drug discovery / pharmacology / in vivo: usually skip
- Biologics CMC (HPLC/CE/protein): skip
- cGMP / QA / manufacturing: skip
- Computational biology: skip
- Organic/process chemistry: skip

CRITICAL — MATCHING SKILLS RULES:
- matching_skills MUST ONLY list skills that are EXPLICITLY MENTIONED in the job description or title
- NEVER infer, assume, or project candidate skills onto the job — if the job does not mention qPCR, do NOT list qPCR as a matching skill
- If the description is a generic company blurb with no specific job requirements, treat it as title-only
- "matching_skills" = skills the JOB ASKS FOR that the candidate HAS (intersection of job requirements and candidate profile)
- "missing_skills" = skills the JOB ASKS FOR that the candidate LACKS (gap between job requirements and candidate profile)
- IMPORTANT: missing_skills must ONLY contain skills REQUIRED BY THE JOB that the candidate does not have. Do NOT list candidate skills that the job doesn't mention — if the job doesn't require qPCR, do NOT put qPCR in missing_skills. If the job has no specific technical requirements, missing_skills can be empty.

SCORING RULES:
- Score 0-100 based on overlap between candidate skills and STATED job requirements
- fit_bucket: strong (70+), moderate (55-69), weak (40-54), poor (<40)
- recommendation: apply (60+), maybe (45-59), skip (<45)
- Be precise about what matches and what's missing
- TITLE-ONLY or THIN DESCRIPTION: If the job has no description or only a brief/generic snippet with no specific requirements listed, cap the score at 50 maximum. Note "limited info — title-only assessment" in reasoning. Only match on what the title explicitly indicates (e.g., "Bioanalytical Scientist" matches bioanalytical domain, but do NOT list specific techniques unless the posting mentions them)

OUTPUT FORMAT — respond with ONLY valid JSON, no markdown:
{{
  "fit_score": <int 0-100>,
  "fit_bucket": "<strong|moderate|weak|poor>",
  "recommendation": "<apply|maybe|skip>",
  "matching_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
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

IMPORTANT: Only list matching_skills that are EXPLICITLY mentioned or required in the description above. Do NOT infer skills based on the job title or company alone."""
    else:
        return f"""Evaluate this job for candidate fit (TITLE ONLY — no substantive description available, cap score at 50 max):

TITLE: {title}
COMPANY: {company}

NOTE: No specific job requirements are available. Score based ONLY on what the title explicitly indicates. Do NOT infer or project specific techniques (qPCR, flow cytometry, etc.) unless the title mentions them. matching_skills should only contain broad domain matches evident from the title."""


async def evaluate_single_job(
    client,
    model: str,
    system_prompt: str,
    title: str,
    company: str,
    description: str,
    description_max_chars: int,
) -> dict:
    """Evaluate a single job using the Claude API. Returns evaluation dict."""
    description_available = bool(description and str(description).strip()
                                  and str(description).lower() != "nan")
    truncated_desc = ""
    if description_available:
        truncated_desc = str(description).strip()[:description_max_chars]

    user_prompt = _build_user_prompt(title, company, truncated_desc, description_available)

    try:
        response = await asyncio.to_thread(
            client.messages.create,
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw_text = response.content[0].text.strip()
        # Strip markdown fences if present
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

        result = json.loads(raw_text)

        # Validate required fields
        required = ["fit_score", "fit_bucket", "recommendation",
                     "matching_skills", "missing_skills", "domain_match", "reasoning"]
        for field in required:
            if field not in result:
                result[field] = "" if field in ("domain_match", "reasoning") else ([] if "skills" in field else 0)

        # Ensure score is int
        result["fit_score"] = int(result.get("fit_score", 0))

        result["description_available"] = description_available
        result["input_tokens"] = response.usage.input_tokens
        result["output_tokens"] = response.usage.output_tokens
        return result

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error for '{title}': {e}")
        return _error_result(title, f"JSON parse error: {e}", description_available)
    except Exception as e:
        logger.warning(f"API error for '{title}': {e}")
        return _error_result(title, str(e), description_available)


def _error_result(title: str, error: str, description_available: bool) -> dict:
    return {
        "fit_score": 0,
        "fit_bucket": "error",
        "recommendation": "skip",
        "matching_skills": [],
        "missing_skills": [],
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
) -> list[dict]:
    """Evaluate a batch of jobs using the Claude API with rate limiting.

    Args:
        jobs_df: DataFrame with columns: job_url, title, company, description
        config: EvaluationConfig with API settings
        progress_callback: optional callable(completed, total) for progress updates

    Returns:
        List of dicts, each with job_url + evaluation fields
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required. Install with: pip install anthropic")

    api_key = config.get_api_key()
    if not api_key:
        raise ValueError("No Anthropic API key found. Set ANTHROPIC_API_KEY env var or anthropic_api_key in config.yaml")

    client = anthropic.Anthropic(api_key=api_key)
    profile = load_resume_profile(config)
    system_prompt = _build_system_prompt(profile)

    results = []
    total = len(jobs_df)
    semaphore = asyncio.Semaphore(config.max_concurrent)
    completed = 0

    async def eval_with_rate_limit(row):
        nonlocal completed
        async with semaphore:
            result = await evaluate_single_job(
                client=client,
                model=config.model,
                system_prompt=system_prompt,
                title=str(row.get("title", "")),
                company=str(row.get("company", "")),
                description=str(row.get("description", "")),
                description_max_chars=config.description_max_chars,
            )
            result["job_url"] = row["job_url"]
            result["evaluated_timestamp"] = pd.Timestamp.now().isoformat()
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

    return results


def run_evaluation_pipeline(
    jobs_df: pd.DataFrame,
    config: EvaluationConfig,
    store,
    prefilter_only: bool = False,
    re_evaluate: bool = False,
    progress_callback=None,
) -> dict:
    """Run the full evaluation pipeline: pre-filter → skip evaluated → Claude API → persist.

    Args:
        jobs_df: Full DataFrame of jobs to evaluate
        config: EvaluationConfig
        store: EvaluationStore instance
        prefilter_only: If True, only run Stage 1 (no API calls)
        re_evaluate: If True, re-evaluate even if already scored
        progress_callback: optional callable(completed, total)

    Returns:
        Summary dict with counts
    """
    total = len(jobs_df)
    logger.info(f"Evaluation pipeline starting with {total} jobs")

    # Stage 1: Pre-filter
    skip_count = 0
    boost_count = 0
    evaluate_jobs = []

    for _, row in jobs_df.iterrows():
        job_url = row.get("job_url", "")
        title = str(row.get("title", ""))
        description = str(row.get("description", ""))

        pf = prefilter_job(title, description)

        if pf.action == "skip":
            skip_count += 1
            # Store the skip result
            store.add_evaluation(job_url, {
                "fit_score": 0,
                "fit_bucket": "prefilter_skip",
                "recommendation": "skip",
                "matching_skills": [],
                "missing_skills": [],
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
        return {
            "total": total,
            "prefilter_skipped": skip_count,
            "boosted": boost_count,
            "to_evaluate": len(evaluate_jobs),
            "evaluated": 0,
            "already_evaluated": 0,
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
        return {
            "total": total,
            "prefilter_skipped": skip_count,
            "boosted": boost_count,
            "to_evaluate": len(evaluate_jobs),
            "evaluated": 0,
            "already_evaluated": already_evaluated,
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

    # Stage 2: Claude API evaluation
    results = asyncio.run(evaluate_batch(to_api_df, config, progress_callback))

    # Persist results
    for result in results:
        job_url = result.pop("job_url", "")
        if job_url:
            store.add_evaluation(job_url, result)

    store.save()

    return {
        "total": total,
        "prefilter_skipped": skip_count,
        "boosted": boost_count,
        "to_evaluate": len(evaluate_jobs),
        "evaluated": len(results),
        "already_evaluated": already_evaluated,
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

    # Haiku pricing: $0.80/M input, $4.00/M output (as of 2025)
    avg_input_tokens = 2500
    avg_output_tokens = 300
    input_cost = (evaluate_count * avg_input_tokens / 1_000_000) * 0.80
    output_cost = (evaluate_count * avg_output_tokens / 1_000_000) * 4.00
    total_cost = input_cost + output_cost

    return {
        "total_jobs": total,
        "prefilter_skip": skip_count,
        "already_evaluated": already_evaluated,
        "to_evaluate": evaluate_count,
        "estimated_input_tokens": evaluate_count * avg_input_tokens,
        "estimated_output_tokens": evaluate_count * avg_output_tokens,
        "estimated_cost_usd": round(total_cost, 4),
    }
```

---

## File: src/eval_persistence.py

```python
"""Persistent storage for job evaluation results (evaluations.json)."""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import EvaluationConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)


class EvaluationStore:
    """Manages evaluations.json — keyed by job_url to prevent re-evaluation."""

    def __init__(self, config: EvaluationConfig):
        self.path = PROJECT_ROOT / config.evaluations_store
        self.model = config.model
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load evaluations: {e}")
        return {"metadata": {}, "evaluations": {}}

    def save(self):
        """Persist evaluations to disk."""
        self._data["metadata"]["last_evaluation_run"] = datetime.now().isoformat()
        self._data["metadata"]["total_evaluated"] = len(self._data["evaluations"])
        self._data["metadata"]["model_used"] = self.model
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)
        logger.info(f"Evaluations saved: {self.path} ({len(self._data['evaluations'])} entries)")

    def is_evaluated(self, job_url: str) -> bool:
        return job_url in self._data.get("evaluations", {})

    def add_evaluation(self, job_url: str, evaluation: dict):
        if "evaluations" not in self._data:
            self._data["evaluations"] = {}
        self._data["evaluations"][job_url] = evaluation

    def get_evaluation(self, job_url: str) -> dict | None:
        return self._data.get("evaluations", {}).get(job_url)

    def get_all_evaluations(self) -> dict:
        return self._data.get("evaluations", {})

    def get_unevaluated(self, job_urls: list[str]) -> list[str]:
        """Return job_urls that haven't been evaluated yet."""
        evaluated = self._data.get("evaluations", {})
        return [url for url in job_urls if url not in evaluated]

    def merge_to_dataframe(self, jobs_df: pd.DataFrame) -> pd.DataFrame:
        """Merge evaluation results into a jobs DataFrame.

        Joins on job_url, adding evaluation columns to the DataFrame.
        Only returns rows that have evaluations.
        """
        evals = self._data.get("evaluations", {})
        if not evals:
            return pd.DataFrame()

        eval_records = []
        for url, ev in evals.items():
            record = {"job_url": url}
            record.update(ev)
            # Convert list fields to comma-joined strings for display
            if isinstance(record.get("matching_skills"), list):
                record["matching_skills"] = ", ".join(record["matching_skills"])
            if isinstance(record.get("missing_skills"), list):
                record["missing_skills"] = ", ".join(record["missing_skills"])
            eval_records.append(record)

        eval_df = pd.DataFrame(eval_records)

        if jobs_df.empty or eval_df.empty:
            return eval_df

        # Merge with job data to get title, company, location, etc.
        merged = jobs_df.merge(eval_df, on="job_url", how="inner")
        return merged

    def export_csv(self, jobs_df: pd.DataFrame, output_path: str, min_score: int = 0) -> Path:
        """Export evaluation results to CSV, optionally filtered by min score."""
        merged = self.merge_to_dataframe(jobs_df)
        if merged.empty:
            logger.warning("No evaluation data to export")
            return Path(output_path)

        if min_score > 0:
            merged = merged[merged["fit_score"] >= min_score]

        merged = merged.sort_values("fit_score", ascending=False)
        path = Path(output_path)
        merged.to_csv(path, index=False)
        logger.info(f"Exported {len(merged)} evaluations to {path}")
        return path

    def summary(self) -> dict:
        """Return summary statistics of evaluations."""
        evals = self._data.get("evaluations", {})
        if not evals:
            return {"total": 0}

        scores = []
        buckets = {"strong": 0, "moderate": 0, "weak": 0, "poor": 0,
                    "prefilter_skip": 0, "error": 0}
        recommendations = {"apply": 0, "maybe": 0, "skip": 0}
        total_input_tokens = 0
        total_output_tokens = 0

        for ev in evals.values():
            score = ev.get("fit_score", 0)
            bucket = ev.get("fit_bucket", "poor")
            rec = ev.get("recommendation", "skip")
            scores.append(score)
            buckets[bucket] = buckets.get(bucket, 0) + 1
            recommendations[rec] = recommendations.get(rec, 0) + 1
            total_input_tokens += ev.get("input_tokens", 0)
            total_output_tokens += ev.get("output_tokens", 0)

        api_evaluated = [s for s, ev in zip(scores, evals.values())
                         if ev.get("fit_bucket") != "prefilter_skip"]

        return {
            "total": len(evals),
            "buckets": buckets,
            "recommendations": recommendations,
            "avg_score": round(sum(api_evaluated) / len(api_evaluated), 1) if api_evaluated else 0,
            "max_score": max(scores) if scores else 0,
            "min_score_api": min(api_evaluated) if api_evaluated else 0,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "metadata": self._data.get("metadata", {}),
        }
```

---

## File: src/description_fetcher.py

```python
"""Fetch job descriptions from URLs for jobs missing descriptions.

Used by the evaluation pipeline to enrich title-only or boilerplate-only
job listings before sending to Claude API for scoring.
"""

import logging
import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

TIMEOUT = 15  # seconds per request

# CSS selectors to try, in priority order (most specific first)
_JOB_DESC_SELECTORS = [
    # Common ATS platforms
    '[class*="job-description"]',
    '[class*="job_description"]',
    '[class*="jobDescription"]',
    '[id*="job-description"]',
    '[id*="job_description"]',
    '[id*="jobDescription"]',
    '[class*="job-details"]',
    '[class*="job_details"]',
    '[class*="jobDetails"]',
    # Greenhouse
    "#content .body",
    "#content",
    # Lever
    ".posting-page .content",
    ".posting-page",
    # Workday
    '[class*="job-posting"]',
    '[data-automation-id="jobPostingDescription"]',
    # iCIMS
    ".iCIMS_JobContent",
    # LinkedIn public job page
    ".description__text",
    ".show-more-less-html__markup",
    # Generic patterns
    '[class*="description"]',
    '[role="main"]',
    "main",
    "article",
]


def _extract_from_html(html: str) -> str:
    """Extract job description text from HTML page."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise elements
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript", "iframe"]):
        tag.decompose()

    # Try selectors in priority order
    for selector in _JOB_DESC_SELECTORS:
        try:
            elements = soup.select(selector)
        except Exception:
            continue
        for elem in elements:
            text = elem.get_text(separator="\n", strip=True)
            if len(text) > 200:
                return text[:10000]

    # Fallback: body text
    body = soup.find("body")
    if body:
        text = body.get_text(separator="\n", strip=True)
        if len(text) > 200:
            return text[:10000]

    return ""


def _is_valid_url(url: str) -> bool:
    """Check if a URL string is non-empty and not NaN."""
    if not url:
        return False
    url = str(url).strip()
    return bool(url) and url.lower() != "nan" and url.startswith("http")


def fetch_description(
    job_url: str,
    job_url_direct: str = "",
) -> Optional[str]:
    """Try to fetch job description from URL(s).

    Tries direct URL first (company career page), then main job URL.
    Returns extracted description text, or None if fetch failed.
    """
    urls_to_try = []

    # Direct URL first (company ATS page — more likely to be scrapable)
    if _is_valid_url(job_url_direct):
        urls_to_try.append(str(job_url_direct).strip())

    # Main job URL second
    if _is_valid_url(job_url):
        urls_to_try.append(str(job_url).strip())

    for url in urls_to_try:
        try:
            resp = requests.get(
                url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True,
            )
            if resp.status_code == 200:
                text = _extract_from_html(resp.text)
                if text and len(text) > 200:
                    logger.debug(f"Fetched description ({len(text)} chars) from {url}")
                    return text
        except requests.exceptions.Timeout:
            logger.debug(f"Timeout fetching {url}")
        except requests.exceptions.ConnectionError:
            logger.debug(f"Connection error fetching {url}")
        except Exception as e:
            logger.debug(f"Error fetching {url}: {e}")

    return None


def fetch_missing_descriptions(
    jobs_df,
    needs_fetch_mask,
    delay: float = 1.5,
    progress_callback=None,
) -> dict:
    """Fetch descriptions for jobs that need them.

    Args:
        jobs_df: DataFrame with job_url, job_url_direct, description columns
        needs_fetch_mask: Boolean mask of rows that need fetching
        delay: Seconds between requests
        progress_callback: optional callable(fetched, total, succeeded)

    Returns:
        Dict mapping DataFrame index -> fetched description text
    """
    indices = jobs_df.index[needs_fetch_mask].tolist()
    total = len(indices)
    fetched = {}
    succeeded = 0

    logger.info(f"Fetching descriptions for {total} jobs with missing/thin descriptions...")

    for i, idx in enumerate(indices):
        row = jobs_df.loc[idx]
        job_url = str(row.get("job_url", ""))
        job_url_direct = str(row.get("job_url_direct", ""))

        desc = fetch_description(job_url, job_url_direct)
        if desc:
            fetched[idx] = desc
            succeeded += 1

        if progress_callback:
            progress_callback(i + 1, total, succeeded)

        # Rate limit
        if i < total - 1:
            time.sleep(delay)

    logger.info(f"Description fetch complete: {succeeded}/{total} succeeded")
    return fetched
```

---

## File: src/resume_profile.py

```python
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
```

---

## File: data/resume_profile.json

```json
{
  "name": "BioTechNerd",
  "years_experience": 18,
  "target_level": ["Senior Scientist", "Principal Scientist", "Associate Director", "Director"],
  "career_anchors": {
    "CRL": {
      "period": "2021-2025",
      "title": "Principal Research Scientist / Senior Research Scientist",
      "key_skills": [
        "qPCR/RT-qPCR/ddPCR method development and GLP validation",
        "AAV and lentiviral vector characterization",
        "Biodistribution, shedding, persistence assays for tox/PK/TK",
        "LNP-based nucleic acid therapy bioanalysis",
        "Clinical shedding assay (human matrices)",
        "PI for 25+ regulatory studies (IND/BLA support)",
        "Method transfer to external CROs",
        "Computer System Validation (21 CFR Part 11)",
        "Team leadership (4 scientists)",
        "$2.5M annual revenue generation",
        "384-well qPCR throughput optimization",
        "Excel/VBA automation tools"
      ]
    },
    "DLS": {
      "period": "2019-2021",
      "title": "Senior Research Scientist",
      "key_skills": [
        "12+ color flow cytometry (T-cell subsets, macrophage M1/M2, TILs)",
        "Pan-T isolation, CD3/CD28 activation, IL-2 expansion",
        "Cell-based potency (luciferase 3D-Glo Max, dose-response IC50)",
        "Cancer organoid biobank ($5M: CRC, lung, breast)",
        "Multiplex cytokine: Bio-Plex/Luminex, MACSPlex, LegendPlex",
        "10X Genomics scRNA-seq",
        "Primary cell isolation (T, B, NK, macrophages, monocytes, DCs, CAFs)",
        "96-well drug screening with clinical compounds"
      ]
    },
    "Stephenson": {
      "period": "2013-2019",
      "title": "Staff Scientist / Postdoctoral Fellow",
      "key_skills": [
        "RT-qPCR gene expression analysis",
        "Seahorse XFe96 metabolic profiling",
        "cfDNA liquid biopsy for ovarian cancer",
        "High-content imaging (Operetta platform)",
        "CRISPR/Cas9 gene editing",
        "Xenograft in vivo studies",
        "8 publications (Cancer Research, Cancer Letters)"
      ]
    },
    "Moffitt": {
      "period": "2007-2013",
      "title": "Postdoctoral Fellow",
      "key_skills": [
        "HDAC biology (HDAC10/MSH2, HDAC6/MSH2)",
        "Site-directed mutagenesis, molecular cloning",
        "$110K research grant as PI",
        "4 publications (Molecular Cell, J Biol Chem)"
      ]
    }
  },
  "core_technical_platforms": [
    "qPCR/RT-qPCR (QuantStudio 7 Flex, Bio-Rad CFX)",
    "ddPCR (QX200)",
    "Flow cytometry (12+ color, FlowJo, FACS sorting)",
    "Luminex/Bio-Plex multiplex immunoassays",
    "Cell-based potency assays (luciferase reporter)",
    "Organoid culture (Lgr5+ stem cells, Matrigel 3D)",
    "10X Genomics scRNA-seq",
    "Seahorse XFe96 metabolic analyzer",
    "High-content imaging (Operetta)",
    "ELISA (sandwich, competitive, capsid)",
    "Maxwell RSC automated extraction",
    "Liquid handlers (Janus, Pipetmax)"
  ],
  "regulatory_framework": [
    "GLP method validation (ICH M10 bioanalytical, ICH Q2(R1) analytical)",
    "21 CFR Part 11 Computer System Validation",
    "GLP study direction and phase report authoring",
    "QA audit participation, deviation/CAPA assessment",
    "OOS/OOT root cause investigation",
    "SOP authoring and harmonization across global sites"
  ],
  "strongest_fit_domains": [
    "CRO management / bioanalytical outsourcing",
    "Gene therapy bioanalytical (AAV, lentiviral, LNP)",
    "Cell therapy analytical (CAR-T, lentiviral)",
    "Flow cytometry / immunophenotyping",
    "Biomarker / translational biomarker",
    "Method validation (qPCR/ddPCR-based)"
  ],
  "moderate_fit_domains": [
    "Cell biology / in vitro assay development",
    "Molecular diagnostics / qPCR-focused",
    "NAMs / in vitro toxicology / 3D models",
    "Oncology target validation / cancer biology",
    "IVD / analytical validation",
    "GPCR signaling-related roles",
    "GMP AS&T / QC analytical (with qPCR/method transfer overlap)"
  ],
  "skip_domains": [
    "Drug discovery / pharmacology / in vivo",
    "Biologics CMC analytical (HPLC/CE/protein-focused)",
    "cGMP / QA / manufacturing",
    "Computational biology (Python/R/ML)",
    "Organic chemistry / process chemistry"
  ],
  "never_claim": [
    "JMP software",
    "LC-MS / small molecule bioanalysis",
    "Direct IND section writing",
    "Tim-3 or LAG-3 markers",
    "MSD platform",
    "ADA / anti-drug antibody assays",
    "ELISpot",
    "Bioreactor operation",
    "R or Python programming",
    "cGMP manufacturing / cleanroom",
    "CHO cell line development",
    "RNA in vitro transcription / mRNA vaccine technology",
    "iPSC directed differentiation"
  ]
}
```

---

## File: tests/test_evaluator.py

```python
"""Tests for the job evaluation pre-filter (Stage 1)."""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluator import prefilter_job


class TestSkipTitles:
    """Titles that should be auto-skipped."""

    def test_skip_vp(self):
        result = prefilter_job("VP of Research Operations")
        assert result.action == "skip"

    def test_skip_svp(self):
        result = prefilter_job("SVP Bioanalytical Sciences")
        assert result.action == "skip"

    def test_skip_hplc_scientist(self):
        result = prefilter_job("HPLC Scientist - Analytical Development")
        assert result.action == "skip"

    def test_skip_lcms(self):
        result = prefilter_job("LC-MS Bioanalytical Scientist")
        assert result.action == "skip"

    def test_skip_qc_analyst(self):
        result = prefilter_job("QC Analyst - Release Testing")
        assert result.action == "skip"

    def test_skip_research_associate(self):
        result = prefilter_job("Research Associate - Cell Biology")
        assert result.action == "skip"

    def test_skip_lab_technician(self):
        result = prefilter_job("Lab Technician - Molecular Biology")
        assert result.action == "skip"

    def test_skip_bioinformatics(self):
        result = prefilter_job("Bioinformatics Scientist")
        assert result.action == "skip"

    def test_skip_computational_biologist(self):
        result = prefilter_job("Computational Biologist")
        assert result.action == "skip"

    def test_skip_data_scientist(self):
        result = prefilter_job("Data Scientist - Genomics")
        assert result.action == "skip"

    def test_skip_organic_chemist(self):
        result = prefilter_job("Organic Chemist - Process Development")
        assert result.action == "skip"

    def test_skip_medicinal_chemist(self):
        result = prefilter_job("Medicinal Chemist")
        assert result.action == "skip"

    def test_skip_postdoc(self):
        result = prefilter_job("Postdoctoral Fellow - Cancer Biology")
        assert result.action == "skip"

    def test_skip_intern(self):
        result = prefilter_job("Intern - Molecular Biology Summer 2026")
        assert result.action == "skip"

    def test_skip_sales(self):
        result = prefilter_job("Sales Representative - Pharma")
        assert result.action == "skip"

    def test_skip_manufacturing_technician(self):
        result = prefilter_job("Manufacturing Technician - Cell Therapy")
        assert result.action == "skip"

    def test_skip_protein_scientist(self):
        result = prefilter_job("Protein Scientist - Biologics")
        assert result.action == "skip"

    def test_skip_formulation_scientist(self):
        result = prefilter_job("Formulation Scientist - Drug Product")
        assert result.action == "skip"

    def test_skip_msl(self):
        result = prefilter_job("Medical Science Liaison - Oncology")
        assert result.action == "skip"

    def test_skip_project_manager(self):
        result = prefilter_job("Project Manager - Clinical Operations")
        assert result.action == "skip"

    def test_skip_analytical_chemist(self):
        result = prefilter_job("Analytical Chemist - QC")
        assert result.action == "skip"


class TestPassTitles:
    """Titles that should NOT be skipped."""

    def test_pass_senior_scientist(self):
        result = prefilter_job("Senior Scientist - Bioanalytical")
        assert result.action != "skip"

    def test_pass_principal_scientist(self):
        result = prefilter_job("Principal Scientist - Gene Therapy")
        assert result.action != "skip"

    def test_pass_associate_director(self):
        result = prefilter_job("Associate Director - Bioanalytical Sciences")
        assert result.action != "skip"

    def test_pass_director_molecular_biology(self):
        result = prefilter_job("Director - Molecular Biology")
        assert result.action != "skip"

    def test_pass_scientist_flow_cytometry(self):
        result = prefilter_job("Scientist - Flow Cytometry")
        assert result.action != "skip"

    def test_pass_method_validation(self):
        result = prefilter_job("Scientist - Method Validation")
        assert result.action != "skip"

    def test_pass_cell_therapy_scientist(self):
        result = prefilter_job("Scientist - Cell Therapy Analytics")
        assert result.action != "skip"

    def test_pass_study_director(self):
        result = prefilter_job("Study Director - Bioanalytical")
        assert result.action != "skip"

    def test_pass_staff_scientist(self):
        result = prefilter_job("Staff Scientist - Oncology Research")
        assert result.action != "skip"


class TestBoostPatterns:
    """Jobs that should be boosted (evaluated with priority)."""

    def test_boost_bioanalytical_title(self):
        result = prefilter_job("Senior Scientist - Bioanalytical")
        assert result.action == "boost"

    def test_boost_gene_therapy(self):
        result = prefilter_job("Scientist - Gene Therapy Bioanalytical")
        assert result.action == "boost"

    def test_boost_car_t(self):
        result = prefilter_job("Scientist - CAR-T Analytics")
        assert result.action == "boost"

    def test_boost_flow_cytometry(self):
        result = prefilter_job("Scientist - Flow Cytometry")
        assert result.action == "boost"

    def test_boost_method_validation(self):
        result = prefilter_job("Scientist - Method Validation")
        assert result.action == "boost"

    def test_boost_qpcr_in_description(self):
        result = prefilter_job("Senior Scientist", "Experience with qPCR and ddPCR methods")
        assert result.action == "boost"

    def test_boost_viral_vector(self):
        result = prefilter_job("Scientist", "Work with AAV viral vector characterization")
        assert result.action == "boost"

    def test_boost_glp(self):
        result = prefilter_job("Scientist", "GLP method validation for biodistribution studies")
        assert result.action == "boost"


class TestRescueLogic:
    """Jobs that match skip description patterns but are rescued."""

    def test_rescue_hplc_but_qpcr(self):
        """Job mentions HPLC extensively but also requires qPCR — should not skip."""
        result = prefilter_job(
            "Senior Scientist - Analytical",
            "Must have extensive experience with HPLC and qPCR methods for viral vector characterization"
        )
        assert result.action != "skip"

    def test_rescue_cho_but_gene_therapy(self):
        """Description mentions CHO cells but also gene therapy — should not skip."""
        result = prefilter_job(
            "Scientist - Analytical Development",
            "CHO cell line development for gene therapy products. Experience with ddPCR."
        )
        assert result.action != "skip"

    def test_rescue_ada_but_flow_cytometry(self):
        """Description mentions ADA assay but also flow cytometry — should not skip."""
        result = prefilter_job(
            "Scientist",
            "ADA assay development and flow cytometry for cell therapy monitoring"
        )
        assert result.action != "skip"

    def test_no_rescue_pure_hplc(self):
        """Pure HPLC role with no rescue keywords — should skip."""
        result = prefilter_job(
            "Senior Scientist",
            "Must have extensive experience with HPLC for protein characterization. SEC-HPLC, IEX."
        )
        assert result.action == "skip"

    def test_no_rescue_pure_bioreactor(self):
        """Pure bioreactor role — should skip."""
        result = prefilter_job(
            "Scientist",
            "Bioreactor operation and cell culture scale-up for manufacturing."
        )
        assert result.action == "skip"


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_title(self):
        result = prefilter_job("")
        assert result.action == "evaluate"

    def test_none_title(self):
        result = prefilter_job(None)
        assert result.action == "evaluate"

    def test_empty_description(self):
        result = prefilter_job("Scientist", "")
        assert result.action == "evaluate"

    def test_case_insensitive_skip(self):
        result = prefilter_job("BIOINFORMATICS SCIENTIST")
        assert result.action == "skip"

    def test_case_insensitive_boost(self):
        result = prefilter_job("scientist - BIOANALYTICAL DEVELOPMENT")
        assert result.action == "boost"
```

---

*End of reference.*
