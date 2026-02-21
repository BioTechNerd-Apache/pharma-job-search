# Pharma/Biotech Job Search Tool

## Project Overview
Personal Python CLI + Streamlit dashboard for Dr. Ranga Radhakrishnan that aggregates pharma/biotech job listings from 5 sources (Indeed, LinkedIn via JobSpy; USAJobs, Adzuna, Jooble via APIs), with an AI-powered evaluation pipeline that scores jobs against the candidate's resume profile using Claude Haiku. Runs locally on macOS, data syncs via iCloud.

## Architecture
```
job_search.py              # CLI entry point (argparse) — search, evaluate, dashboard
config.yaml                # Search terms, API keys, filters, synonyms, evaluation config
requirements.txt           # Python dependencies
src/
  config.py                # YAML loader + dataclasses (AppConfig, EvaluationConfig, etc.)
  aggregator.py            # Orchestrator: ThreadPoolExecutor(5), normalize, filter
  dedup.py                 # 3-layer dedup: URL, fuzzy (company+title+state), cross-source
  exporter.py              # CSV/Excel merge-on-save to single master file
  dashboard.py             # Streamlit UI with AG Grid — 2 tabs: Job Listings + Evaluations
  scraper_jobspy.py        # Indeed/LinkedIn via python-jobspy (run as separate parallel tasks)
  scraper_usajobs.py       # USAJobs REST API
  scraper_adzuna.py        # Adzuna REST API
  scraper_jooble.py        # Jooble REST API (key currently blank)
  evaluator.py             # 2-stage evaluation: rule-based pre-filter + Claude API scoring
  eval_persistence.py      # EvaluationStore: JSON persistence for evaluations (keyed by job_url)
  description_fetcher.py   # Fetches full job descriptions from URLs via HTML scraping
  resume_profile.py        # Resume profile loader + validator
data/
  pharma_jobs.csv          # Master rolling CSV
  pharma_jobs.xlsx         # Master rolling Excel
  pharma_jobs_raw.csv      # Raw pre-filter/dedup data (for --reprocess)
  reviewed.json            # Review tracking {job_url: "YYYY-MM-DD HH:MM"}
  evaluations.json         # AI evaluation results {metadata, evaluations: {job_url: scores}}
  resume_profile.json      # Structured candidate profile for AI evaluation
tests/
  test_evaluator.py        # Tests for Stage 1 pre-filter (skip/pass/boost/rescue logic)
```

## How to Run
```bash
# Search
python job_search.py                    # Full search
python job_search.py --days 1           # Last 24hrs
python job_search.py --reprocess        # Re-filter/dedup from raw data (no scraping)
python job_search.py --web              # Dashboard only

# Evaluation
python job_search.py --evaluate               # Scrape + evaluate new jobs
python job_search.py --evaluate-only           # Evaluate without scraping
python job_search.py --eval-days 3            # Evaluate jobs from last 3 days
python job_search.py --eval-all               # Evaluate all unevaluated jobs
python job_search.py --eval-prefilter-only    # Stage 1 only (no API calls)
python job_search.py --eval-dry-run           # Show count + cost estimate
python job_search.py --eval-summary           # Show evaluation stats
python job_search.py --eval-export results.csv --eval-min-score 60  # Export results
python job_search.py --re-evaluate            # Force re-evaluation of scored jobs

# Or double-click: "ranga's Job Search.command"
```

## Key Design Decisions

- **Discipline filter**: Both include AND exclude filters match job **title only** (not description). This prevents false positives from keywords in descriptions.
- **Reviewed-job preservation**: Reviewed jobs get +1000 richness boost in dedup so they always survive when duplicates are merged.
- **Dedup layers**: (1) exact URL after stripping tracking params, (2) fuzzy key = normalized(company|title|state), (3) cross-source = same title + company similarity check including known job board detection.
- **Progressive save**: Each scraper saves to master CSV immediately on completion (thread-safe via locks), not waiting for all scrapers to finish.
- **Repost detection**: All date_posted values from duplicate groups are collected and stored in reposted_date column.
- **iCloud sync**: The .command launcher uses `brctl download` to force iCloud to download data files before starting the dashboard.
- **Evaluation pipeline**: 2-stage design — Stage 1 (rule-based pre-filter) skips obvious mismatches without API cost, Stage 2 (Claude Haiku) scores remaining jobs against resume profile.
- **Description enrichment**: Stage 1.5 fetches full job descriptions from URLs for jobs with missing/thin descriptions before sending to Claude API.
- **Evaluation persistence**: Results stored in evaluations.json keyed by job_url to prevent re-evaluation; `--re-evaluate` flag overrides this.

## Evaluation Pipeline Detail

### Stage 1: Rule-based Pre-filter (evaluator.py)
- **Skip patterns**: ~50 regex patterns on title (VP, QC analyst, HPLC scientist, data scientist, postdoc, etc.) and ~15 on description (extensive HPLC, bioreactor operation, CHO cells, etc.)
- **Rescue patterns**: ~15 patterns (qPCR, flow cytometry, gene therapy, AAV, CAR-T, GLP, etc.) — if a job matches both skip AND rescue, it is NOT skipped
- **Boost patterns**: ~15 patterns (bioanalytical, gene/cell therapy, CAR-T, flow cytometry, method validation, qPCR, etc.) — boosted jobs are evaluated first

### Stage 1.5: Description Fetching (description_fetcher.py)
- Fetches HTML from job URLs and extracts description text via CSS selectors
- Tries direct URL (company career page) first, then main job URL
- Supports Greenhouse, Lever, Workday, iCIMS, LinkedIn, and generic ATS platforms

### Stage 2: Claude API Scoring (evaluator.py)
- Uses Claude Haiku (claude-haiku-4-5-20251001) for cost-effective scoring
- System prompt includes full resume profile + domain calibration table + strict matching rules
- Title-only jobs capped at score 50 max
- Output: fit_score (0-100), fit_bucket (strong/moderate/weak/poor), recommendation (apply/maybe/skip), matching_skills, missing_skills, domain_match, reasoning
- Async evaluation with configurable concurrency and rate limiting

## Important Patterns

- All scrapers return DataFrames with a standard 15-column schema defined in `aggregator.py:OUTPUT_COLUMNS`
- API credentials are in `config.yaml` with environment variable fallbacks (USAJOBS_API_KEY, ADZUNA_APP_ID, ANTHROPIC_API_KEY, etc.)
- Search terms auto-expand via synonym groups defined in config.yaml
- Company name normalization strips 40+ stop words (Inc, LLC, Pharmaceuticals, etc.) for matching
- 50+ known job boards/staffing agencies are recognized for cross-source dedup
- Python 3.14 is hardcoded in the .command launcher
- Dashboard has 2 tabs: "Job Listings" (original grid) and "Evaluation Results" (scored jobs with fit_score color coding)
- Both tabs share the review system (mark/unmark reviewed via AG Grid selection)

## Dependencies
python-jobspy, pandas, openpyxl, streamlit, streamlit-aggrid, pyyaml, requests, beautifulsoup4, anthropic

## Full Reference
See `PROJECT_REFERENCE.md` for complete source code of every file (useful for reconstructing the project in a new session).
