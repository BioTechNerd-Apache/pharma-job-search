# Pharma/Biotech Job Search Tool

## Project Overview
Personal Python CLI + Streamlit dashboard that aggregates pharma/biotech job listings from 5 sources (Indeed, LinkedIn via JobSpy; USAJobs, Adzuna, Jooble via APIs), with an AI-powered evaluation pipeline that scores jobs against the candidate's resume profile using Claude Haiku. Runs locally on macOS, data syncs via iCloud.

## Architecture
```
job_search.py              # CLI entry point (argparse) — search, evaluate, dashboard
config.yaml                # Search terms, API keys, filters, synonyms, evaluation config
requirements.txt           # Python dependencies
src/
  config.py                # YAML loader + dataclasses (AppConfig, EvaluationConfig, etc.)
  aggregator.py            # Orchestrator: shared work queue, per-site semaphores, normalize, filter
  dedup.py                 # 3-layer dedup: URL, fuzzy (company+title+state), cross-source
  exporter.py              # CSV/Excel merge-on-save to single master file
  dashboard.py             # Streamlit UI with AG Grid — 2 tabs: Job Listings + Evaluations
  scraper_jobspy.py        # Indeed/LinkedIn via python-jobspy (single-term + batch modes)
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

# Or double-click: "BioTechNerd-Apache's Job Search.command"
```

## Key Design Decisions

- **Discipline filter**: Both include AND exclude filters match job **title only** (not description). This prevents false positives from keywords in descriptions.
- **Reviewed-job preservation**: Reviewed jobs get +1000 richness boost in dedup so they always survive when duplicates are merged.
- **Dedup layers**: (1) exact URL after stripping tracking params, (2) fuzzy key = normalized(company|title|state), (3) cross-source = same title + company similarity check including known job board detection.
- **Progressive save**: Each completed (site, term) task saves to master CSV immediately (thread-safe via locks), not waiting for all tasks to finish.
- **Repost detection**: All date_posted values from duplicate groups are collected and stored in reposted_date column.
- **iCloud sync**: The .command launcher uses `brctl download` to force iCloud to download data files before starting the dashboard.
- **Evaluation pipeline**: 2-stage design — Stage 1 (rule-based pre-filter) skips obvious mismatches without API cost, Stage 2 (Claude Haiku) scores remaining jobs against resume profile.
- **Description enrichment**: Stage 1.5 fetches full job descriptions from URLs for jobs with missing/thin descriptions before sending to Claude API.
- **Evaluation persistence**: Results stored in evaluations.json keyed by job_url to prevent re-evaluation; `--re-evaluate` flag overrides this.
- **Tiered results_per_site**: Broad search terms (defined in `priority_terms`) request `priority_results_per_site` (200) results; narrow terms use `results_per_site` (100). Prevents relevant jobs from falling outside the top 100 for competitive terms.
- **Parallel work queue**: Instead of 1 worker per site (3 idle after API scrapers finish in ~2 min), all 5 workers share a queue of individual `(site, term)` tasks. API scrapers (USAJobs, Adzuna, Jooble) run first as single tasks, then all workers process interleaved Indeed/LinkedIn `(site, term)` pairs. Per-site `Semaphore(2)` caps concurrency; per-site delay tracking enforces `delay_between_searches` spacing. Roughly 2x faster for the Indeed/LinkedIn portion.

## Scraping Architecture

```
ThreadPoolExecutor(5 workers) pulling from shared work queue:

  [USAJobs, Adzuna, Jooble]                    ← 3 API tasks (finish in seconds)
  [(indeed, term1), (linkedin, term1),          ← 134 JobSpy tasks (67 terms × 2 sites)
   (indeed, term2), (linkedin, term2), ...]        interleaved for balanced load

  Per-site semaphore: max 2 concurrent workers per site
  Per-site delay: 5s between consecutive calls to same site
  Each completed task: normalize → discipline filter → save to master CSV
```

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

- All scrapers return DataFrames with a standard 16-column schema defined in `aggregator.py:OUTPUT_COLUMNS`
- API credentials are in `config.yaml` with environment variable fallbacks (USAJOBS_API_KEY, ADZUNA_APP_ID, ANTHROPIC_API_KEY, etc.)
- Search terms auto-expand via synonym groups defined in config.yaml; ~11 priority terms get 200 results per site (others get 100)
- `scrape_single_term()` handles one (site, term) pair; `scrape_jobs()` retained for backward compatibility
- Company name normalization strips 40+ stop words (Inc, LLC, Pharmaceuticals, etc.) for matching
- 50+ known job boards/staffing agencies are recognized for cross-source dedup
- Python 3.14 is hardcoded in the .command launcher
- Dashboard has 2 tabs: "Job Listings" (original grid) and "Evaluation Results" (scored jobs with fit_score color coding)
- Both tabs share the review system (mark/unmark reviewed via AG Grid selection)

## Dependencies
python-jobspy, pandas, openpyxl, streamlit, streamlit-aggrid, pyyaml, requests, beautifulsoup4, anthropic

## Full Reference
See `docs/PROJECT_REFERENCE.md` for complete source code of every file (useful for reconstructing the project in a new session).
