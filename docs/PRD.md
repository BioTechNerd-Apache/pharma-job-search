# Product Requirements Document (PRD)
# Pharma/Biotech Job Search Aggregator with AI Evaluation

**Version**: 1.0
**Date**: February 2026
**Status**: Released (v1.0)

---

## 1. Executive Summary

A Python CLI + Streamlit dashboard that aggregates pharma/biotech job listings from 5 job boards, deduplicates them across sources, and uses an AI-powered evaluation pipeline to score each job against the user's resume profile. Supports Anthropic (Claude), OpenAI (GPT), and Ollama (free local models). Includes an AI setup wizard that generates all configuration from a resume. Designed for scientists and directors in the pharmaceutical/biotech industry who need to efficiently search across multiple platforms and identify the best-fit roles without manual filtering.

**Core value proposition**: Upload your resume, let AI generate your config, search 5 job boards in parallel, eliminate duplicates, and let AI rank every job against your background — all from a single command.

---

## 2. Problem Statement

Job seekers in pharma/biotech face several challenges:

1. **Fragmented sources**: Relevant jobs are spread across Indeed, LinkedIn, USAJobs (federal), Adzuna, Jooble, and company career pages. No single board has comprehensive coverage.
2. **Noise**: Generic searches for "scientist" return thousands of irrelevant results (data scientists, food scientists, nursing scientists, etc.).
3. **Duplicates**: The same job posted on multiple boards appears multiple times, wasting review time.
4. **Manual evaluation**: Reading each job description to assess fit is time-consuming. A scientist with 100+ new listings per day cannot manually evaluate them all.
5. **Missed reposts**: Jobs that are reposted (indicating urgency or open positions) are hard to detect across platforms.

---

## 3. Solution Overview

### 3.1 System Architecture

```
User (CLI / Dashboard)
        |
        v
  ┌─────────────────────────────────────────────┐
  │              job_search.py (CLI)             │
  │   Parses arguments, routes to workflows      │
  └────────┬──────────┬──────────┬──────────────┘
           |          |          |
     ┌─────v────┐  ┌──v───┐  ┌──v──────────┐
     │  Search  │  │ Eval │  │  Dashboard  │
     │  Mode    │  │ Mode │  │  Mode       │
     └─────┬────┘  └──┬───┘  └─────────────┘
           |          |
           v          v
  ┌─────────────┐  ┌──────────────────────┐
  │ aggregator  │  │ evaluator            │
  │ (5 scrapers │  │ Stage 1: Pre-filter  │
  │  in parallel│  │ Stage 1.5: Desc fetch│
  │  via Thread │  │ Stage 2: Claude API  │
  │  Pool)      │  │                      │
  └──────┬──────┘  └──────────┬───────────┘
         |                    |
         v                    v
  ┌─────────────┐  ┌──────────────────────┐
  │ dedup.py    │  │ eval_persistence.py  │
  │ 3-layer     │  │ evaluations.json     │
  │ dedup       │  │                      │
  └──────┬──────┘  └──────────────────────┘
         |
         v
  ┌─────────────┐
  │ exporter.py │
  │ Master CSV  │
  │ + Excel     │
  └─────────────┘
```

### 3.2 Data Flow

1. **Scrape**: 5 scrapers run in parallel via ThreadPoolExecutor, each querying its job board with configured search terms
2. **Normalize**: Each scraper maps its source-specific fields to a standard 15-column schema
3. **Filter**: Discipline filter keeps only relevant roles by matching title keywords (include/exclude lists)
4. **Deduplicate**: 3-layer dedup eliminates duplicates across sources while tracking repost dates
5. **Save**: Results merge into a single master CSV + Excel file (progressive save as each scraper completes)
6. **Evaluate** (optional): 2-stage AI pipeline scores jobs against the user's resume profile
7. **View**: Streamlit dashboard with AG Grid for browsing, filtering, and reviewing results

---

## 4. Functional Requirements

### 4.1 Multi-Source Job Aggregation

| ID | Requirement | Implementation |
|----|-------------|----------------|
| FR-1.1 | Search Indeed and LinkedIn simultaneously | `scraper_jobspy.py` via python-jobspy library; no API key required |
| FR-1.2 | Search USAJobs (federal government jobs) | `scraper_usajobs.py` via REST API; requires API key + email |
| FR-1.3 | Search Adzuna | `scraper_adzuna.py` via REST API; requires app_id + app_key |
| FR-1.4 | Search Jooble | `scraper_jooble.py` via REST API; requires API key |
| FR-1.5 | All scrapers run in parallel | `aggregator.py` uses `ThreadPoolExecutor(max_workers=5)` |
| FR-1.6 | Progressive save — each scraper saves immediately on completion | Thread-safe via locks in `aggregator.run_search()` |
| FR-1.7 | Configurable search terms with synonym expansion | `config.py` expands synonyms (e.g., "bioanalytical" also searches "bioanalysis", "PK/PD") |
| FR-1.8 | Configurable time window | `--days N` flag; default 7 days |
| FR-1.9 | Configurable results per site | `config.yaml: search.results_per_site` (default 100) |
| FR-1.10 | Rate limiting between searches | `config.yaml: search.delay_between_searches` (default 5s) |

**Standard output schema** (15 columns):
`title, company, location, state, date_posted, reposted_date, days_since_posted, source, job_url, job_url_direct, salary_min, salary_max, is_remote, job_type, description, eval_status`

### 4.2 Discipline Filtering

| ID | Requirement | Implementation |
|----|-------------|----------------|
| FR-2.1 | Include filter — keep jobs matching at least one keyword | `config.yaml: search.filter_include` (~60 keywords) |
| FR-2.2 | Exclude filter — remove jobs matching any keyword | `config.yaml: search.filter_exclude` (~100 keywords) |
| FR-2.3 | Filters match title only (not description) | `aggregator.apply_discipline_filter()` — prevents false positives from description boilerplate |
| FR-2.4 | Case-insensitive matching | All comparisons use `.str.lower()` |

**Design rationale**: Matching on title only is a deliberate decision. Job descriptions often contain generic boilerplate that mentions many disciplines (e.g., "We hire data scientists, research scientists, and engineers"). Filtering on description would produce false positives.

### 4.3 Deduplication

| ID | Requirement | Implementation |
|----|-------------|----------------|
| FR-3.1 | Layer 1: Exact URL dedup after stripping tracking params | `dedup.normalize_url()` strips utm_*, fbclid, gclid, etc. |
| FR-3.2 | Layer 2: Fuzzy dedup on normalized(company\|title\|state) | `dedup.deduplicate()` groups by fuzzy key, keeps richest row |
| FR-3.3 | Layer 3: Cross-source dedup (same title + similar company across boards) | `dedup.companies_match()` checks substring match + 50+ known job board names |
| FR-3.4 | Reviewed jobs always survive dedup | Reviewed rows get +1000 richness score via `dedup.data_richness_score()` |
| FR-3.5 | Repost detection — track all dates from duplicate groups | `dedup.extract_repost_dates()` populates `reposted_date` column |
| FR-3.6 | Company name normalization strips stop words | 40+ stop words (Inc, LLC, Pharmaceuticals, Corp, etc.) |
| FR-3.7 | Known job boards recognized for cross-source matching | 50+ boards (BioSpace, Lever, Glassdoor, Workday, etc.) |

**Data richness scoring**: When duplicates are found, the row with the most useful data survives. Score formula:
- +1 per non-null column
- +1 per 100 chars of description (max +5)
- +1 if has salary, +1 if has direct URL
- +1000 if reviewed by user

### 4.4 Export & Persistence

| ID | Requirement | Implementation |
|----|-------------|----------------|
| FR-4.1 | Single master CSV file (merge-on-save) | `exporter.merge_and_export_csv()` — no timestamped files |
| FR-4.2 | Single master Excel file with auto-adjusted columns | `exporter._export_excel()` via openpyxl |
| FR-4.3 | Preserve eval_status across merges | Status map built before merge, restored after dedup |
| FR-4.4 | Raw data preserved for reprocessing | `pharma_jobs_raw.csv` stores pre-filter/dedup data |
| FR-4.5 | Migration from old timestamped files | `exporter.migrate_old_files()` — one-time merge of `pharma_jobs_*.csv` |

### 4.5 AI-Powered Evaluation Pipeline

#### Stage 1: Rule-Based Pre-Filter (`evaluator.py`)

| ID | Requirement | Implementation |
|----|-------------|----------------|
| FR-5.1 | Skip obvious mismatches by title | ~50 regex patterns in `SKIP_TITLE_PATTERNS` (VP, QC analyst, HPLC scientist, data scientist, postdoc, etc.) |
| FR-5.2 | Skip by description content | ~15 patterns in `SKIP_DESCRIPTION_PATTERNS` (extensive HPLC, bioreactor, CHO cells, etc.) |
| FR-5.3 | Rescue mechanism — override skip if job also matches rescue pattern | ~15 patterns in `RESCUE_PATTERNS` (qPCR, flow cytometry, gene therapy, AAV, CAR-T, GLP) |
| FR-5.4 | Boost high-priority jobs for earlier evaluation | ~15 patterns in `BOOST_PATTERNS` (bioanalytical, gene therapy, method validation) |
| FR-5.5 | Zero API cost for pre-filter | All regex-based, runs locally |

**Pre-filter logic**:
```
IF title matches SKIP_TITLE_PATTERNS:
    IF title also matches RESCUE_PATTERNS → evaluate (rescued)
    ELSE → skip
IF description matches SKIP_DESCRIPTION_PATTERNS:
    IF description also matches RESCUE_PATTERNS → evaluate (rescued)
    ELSE → skip
IF title matches BOOST_PATTERNS → boost (evaluate first)
ELSE → evaluate (normal priority)
```

#### Stage 1.5: Description Enrichment (`description_fetcher.py`)

| ID | Requirement | Implementation |
|----|-------------|----------------|
| FR-5.6 | Fetch full descriptions for jobs with thin/missing text | `fetch_missing_descriptions()` fetches HTML from job URLs |
| FR-5.7 | Support multiple ATS platforms | CSS selectors for Greenhouse, Lever, Workday, iCIMS, LinkedIn, generic |
| FR-5.8 | Try direct URL first, then main URL | `fetch_description()` tries `job_url_direct` before `job_url` |
| FR-5.9 | Rate limiting | 1.5s delay between requests |
| FR-5.10 | Timeout protection | 15s per URL |
| FR-5.11 | Minimum description length | Only accepts descriptions >= 200 chars |

#### Stage 2: AI Scoring (`evaluator.py`)

| ID | Requirement | Implementation |
|----|-------------|----------------|
| FR-5.12 | Score each job against resume profile (0-100) | AI provider (Anthropic/OpenAI/Ollama) via ai_client.py with structured JSON output |
| FR-5.13 | Classify into fit buckets | strong (70+), moderate (55-69), weak (40-54), poor (<40) |
| FR-5.14 | Provide recommendation | apply (60+), maybe (45-59), skip (<45) |
| FR-5.15 | Return matching/missing skills, domain match, reasoning | Structured JSON response parsed from API |
| FR-5.16 | Cap title-only jobs at score 50 | Hard cap enforced in code after AI response: `fit_score` clamped to 50, bucket/recommendation recalculated, `domain_match` prefixed with `"[Title Only] "` |
| FR-5.17 | Async evaluation with configurable concurrency | `asyncio.Semaphore(max_concurrent)` — default 1 |
| FR-5.18 | Retry with exponential backoff on rate limits | 5s, 10s, 20s, 40s... + random jitter |
| FR-5.19 | Fail-fast on authentication errors | Aborts after 3 consecutive auth failures |
| FR-5.20 | Persist results to evaluations.json | `eval_persistence.EvaluationStore` keyed by job_url; incremental saves every 5 jobs during evaluation to prevent data loss on interruption |
| FR-5.21 | Prevent re-evaluation of already-scored jobs | `store.is_evaluated()` check; `--re-evaluate` overrides |
| FR-5.22 | Catch-up evaluation for missed jobs | After date filter, includes any unevaluated jobs from all time |
| FR-5.23 | Cost estimation (dry-run mode) | `estimate_cost()` counts jobs, estimates tokens, calculates USD |
| FR-5.24 | Track token usage | Input and output tokens stored per evaluation |

**System prompt structure**:
- Full resume profile (career anchors, technical platforms, regulatory experience)
- Domain calibration table (strongest fit, moderate fit, skip domains)
- Scoring rules and output format
- "Never claim" list (skills the candidate does NOT have)

**Cost model** (Claude Haiku):
- Input: ~$0.80 / 1M tokens
- Output: ~$4.00 / 1M tokens
- Average per job: ~2,500 input tokens + ~300 output tokens = ~$0.003 USD
- 100 jobs = ~$0.30 | 1,000 jobs = ~$3.00

#### Evaluation Persistence (`eval_persistence.py`)

| ID | Requirement | Implementation |
|----|-------------|----------------|
| FR-5.25 | Store evaluations in JSON keyed by job_url | `evaluations.json` with metadata + evaluations dict |
| FR-5.26 | Track metadata (last run, total evaluated, model) | `save()` updates metadata on each persist |
| FR-5.27 | Summary statistics | `summary()` returns buckets, recommendations, avg score, token totals |
| FR-5.28 | Export evaluations to CSV | `export_csv()` with optional min_score filter |
| FR-5.29 | Merge evaluations with job data | `merge_to_dataframe()` joins on job_url |
| FR-5.30 | Track all evaluated URLs | `get_all_urls()` returns set of all job URLs with entries |

#### Resume Profile (`resume_profile.py`)

| ID | Requirement | Implementation |
|----|-------------|----------------|
| FR-5.31 | Load structured resume from JSON | `load_profile()` reads `data/resume_profile.json` |
| FR-5.32 | Validate required fields | Checks for career_anchors, core_technical_platforms, regulatory_framework, strongest_fit_domains, never_claim |
| FR-5.33 | Profile informs AI scoring | Embedded in Claude system prompt |

**Profile schema**:
```json
{
  "name": "...",
  "years_experience": 12,
  "target_level": ["Senior Scientist", "Associate Director"],
  "career_anchors": { "CompanyName": { "period": "...", "title": "...", "key_skills": [...] } },
  "core_technical_platforms": ["qPCR", "flow cytometry", ...],
  "regulatory_framework": ["GLP", "GMP", ...],
  "strongest_fit_domains": ["gene therapy bioanalytical", ...],
  "moderate_fit_domains": ["cell biology", ...],
  "skip_domains": ["drug discovery", ...],
  "never_claim": ["LC-MS", "ADA assays", ...]
}
```

### 4.6 Interactive Dashboard (`dashboard.py`)

| ID | Requirement | Implementation |
|----|-------------|----------------|
| FR-6.1 | Tab 1: Job Listings — browse all scraped jobs | AG Grid with sortable/filterable columns |
| FR-6.2 | Tab 2: Evaluation Results — view AI-scored jobs | Color-coded fit scores (green 70+, yellow 55-69, orange 40-54, red <40); "Info" column shows description availability status |
| FR-6.3 | Filters: source, state, remote, salary, reposted, reviewed | Sidebar selectboxes and checkboxes |
| FR-6.4 | Review system — mark/unmark jobs as reviewed | Checkbox selection + "Mark Reviewed" button; persists to `reviewed.json` |
| FR-6.5 | Clickable job URLs | AG Grid cell renderer opens links in new tab |
| FR-6.6 | Job code extraction from URLs | LI-xxx (LinkedIn), IN-xxx (Indeed), UJ-xxx (USAJobs), AZ-xxx (Adzuna), JB-xxx (Jooble) |
| FR-6.7 | Summary metrics | Total jobs, source count, state count, reposted count, reviewed count |
| FR-6.8 | Expandable job descriptions | First 50 jobs show description in expandable sections |
| FR-6.9 | Run search from dashboard | Sidebar button triggers search with progress tracking |
| FR-6.10 | Evaluation detail panel | Shows reasoning, domain match, recommendation for selected job |
| FR-6.11 | Pagination | 50 rows per page with AG Grid pagination |
| FR-6.12 | Info column — description availability indicator | Cell renderer shows "⚠️ Title Only" (orange) for `description_available=False`, "✓ Full" (green) otherwise; filterable via AG Grid set filter |
| FR-6.13 | Sidebar filter for title-only jobs | Checkbox filter in Evaluation Results sidebar to isolate title-only evaluations |
| FR-6.14 | Description fetch progress bar | Separate progress bars for Stage 1.5 description fetching and Stage 2 AI evaluation |

### 4.7 CLI Commands

| Command | Purpose |
|---------|---------|
| `python job_search.py --setup resume.pdf` | AI wizard: generates all config from resume |
| `python job_search.py` | Full search (default: last 7 days) |
| `python job_search.py --days N` | Search last N days |
| `python job_search.py --terms "X" "Y"` | Override search terms |
| `python job_search.py --extra-terms "X"` | Append additional search terms |
| `python job_search.py --location "..."` | Override location filter |
| `python job_search.py --sites indeed linkedin` | Override job sites |
| `python job_search.py --reprocess` | Re-run filter/dedup from raw data (no scraping) |
| `python job_search.py --web` | Launch dashboard only |
| `python job_search.py --evaluate` | Search + evaluate new jobs |
| `python job_search.py --evaluate-only` | Evaluate without scraping |
| `python job_search.py --eval-since "2026-02-19 06:30"` | Evaluate jobs since datetime |
| `python job_search.py --eval-days N` | Evaluate jobs from last N days |
| `python job_search.py --eval-all` | Evaluate all unevaluated jobs |
| `python job_search.py --eval-prefilter-only` | Stage 1 only (no API calls) |
| `python job_search.py --eval-dry-run` | Show count + cost estimate |
| `python job_search.py --eval-summary` | Show evaluation statistics |
| `python job_search.py --eval-export FILE` | Export evaluations to CSV |
| `python job_search.py --eval-min-score N` | Filter exports by minimum score |
| `python job_search.py --re-evaluate` | Force re-evaluation of scored jobs |

---

## 5. Non-Functional Requirements

| ID | Requirement | Detail |
|----|-------------|--------|
| NFR-1 | Runs locally | No cloud infrastructure; all data stored on local filesystem |
| NFR-2 | Cross-platform | macOS, Linux, Windows (Python 3.10+) |
| NFR-3 | Minimal dependencies | 8 Python packages (see requirements.txt) |
| NFR-4 | API key security | Keys stored in config.yaml (gitignored) or environment variables; never committed |
| NFR-5 | Graceful degradation | If a scraper fails, others continue; partial results saved |
| NFR-6 | Rate limiting | Configurable delays between searches and API calls |
| NFR-7 | Cost control | Pre-filter eliminates ~50% of jobs before API cost; dry-run mode for estimates |
| NFR-8 | Idempotent evaluation | Jobs evaluated once; re-runs skip already-scored jobs |
| NFR-9 | Data preservation | Master CSV never loses data on merge; reviewed status preserved |
| NFR-10 | Auth failure resilience | Pipeline aborts after 3 consecutive auth errors instead of burning through all jobs |

---

## 6. Configuration & Customization

The tool has **4 layers of customization**, from easiest to most involved:

### Layer 1: Search Terms (`config.yaml`)
- **What**: Keywords sent to job boards
- **Effort**: Edit YAML file
- **Example**: Change `"bioanalytical scientist"` to `"software engineer"`
- **Synonym expansion**: Define groups so one term auto-expands (e.g., `"ML": ["machine learning", "deep learning", "neural network"]`)

### Layer 2: Discipline Filters (`config.yaml`)
- **What**: Post-scrape title keyword filters
- **Effort**: Edit YAML file
- **Include list**: Terms that must appear in title (e.g., `scientist`, `bioanalyt`, `pharma`)
- **Exclude list**: Terms that disqualify a job (e.g., `data scientist`, `nurse`, `sales`)

### Layer 3: Pre-Filter Patterns (`src/evaluator.py`)
- **What**: Regex patterns for rule-based skip/rescue/boost before AI scoring
- **Effort**: Edit Python source code
- **Skip patterns**: ~50 title patterns + ~15 description patterns
- **Rescue patterns**: ~15 patterns that override skips
- **Boost patterns**: ~15 patterns that get priority evaluation
- **Testing**: `--eval-prefilter-only` runs patterns without API cost

### Layer 4: Resume Profile (`data/resume_profile.json`)
- **What**: Structured representation of candidate background for AI scoring
- **Effort**: Edit JSON file
- **Fields**: Career history, technical platforms, regulatory experience, fit domains, anti-skills

---

## 7. Data Model

### 7.1 Master CSV Schema (15 columns + eval_status)

| Column | Type | Description |
|--------|------|-------------|
| title | string | Job title |
| company | string | Company name |
| location | string | Full location string |
| state | string | 2-letter US state code (extracted) |
| date_posted | datetime | Original posting date |
| reposted_date | string | Comma-separated dates from duplicate postings |
| days_since_posted | int | Days since date_posted |
| source | string | Job board (indeed, linkedin, usajobs, adzuna, jooble) |
| job_url | string | Primary job URL |
| job_url_direct | string | Direct employer URL (if available) |
| salary_min | float | Minimum salary (if available) |
| salary_max | float | Maximum salary (if available) |
| is_remote | bool | Remote work flag |
| job_type | string | Full-time, part-time, contract, etc. |
| description | string | Job description text |
| eval_status | string | Evaluation state: "" (pending), "evaluated", "skipped", "prefilter_skipped" |

### 7.2 Evaluations JSON Schema

```json
{
  "metadata": {
    "last_evaluation_run": "2026-02-22T10:30:00",
    "total_evaluated": 1500,
    "model_used": "claude-haiku-4-5-20251001"
  },
  "evaluations": {
    "https://example.com/job/123": {
      "fit_score": 75,
      "fit_bucket": "strong",
      "recommendation": "apply",
      "domain_match": "Gene therapy bioanalytical",
      "matching_skills": ["qPCR", "AAV characterization", "GLP"],
      "missing_skills": ["HPLC experience"],
      "reasoning": "Strong match for bioanalytical scientist role...",
      "description_available": true,
      "input_tokens": 2400,
      "output_tokens": 280,
      "evaluated_timestamp": "2026-02-22T10:30:15"
    }
  }
}
```

### 7.3 Resume Profile JSON Schema

```json
{
  "name": "Candidate Name",
  "years_experience": 12,
  "target_level": ["Senior Scientist", "Associate Director"],
  "career_anchors": {
    "PositionName": {
      "period": "2020-2025",
      "title": "Senior Research Scientist",
      "key_skills": ["skill1", "skill2"]
    }
  },
  "core_technical_platforms": ["platform1", "platform2"],
  "regulatory_framework": ["GLP", "GMP"],
  "strongest_fit_domains": ["domain1"],
  "moderate_fit_domains": ["domain2"],
  "skip_domains": ["domain3"],
  "never_claim": ["skill_I_dont_have"]
}
```

### 7.4 Reviewed JSON Schema

```json
{
  "https://example.com/job/123": "2026-02-22 10:30"
}
```

---

## 8. Module Reference

### 8.1 `job_search.py` — CLI Entry Point
- **parse_args()**: Defines all CLI arguments via argparse
- **launch_dashboard()**: Spawns Streamlit subprocess
- **load_jobs_csv()**: Reads master CSV from disk
- **filter_jobs_by_time()**: Filters DataFrame by --eval-since, --eval-days, or --eval-all; includes catch-up for unevaluated jobs
- **run_evaluation()**: Orchestrates evaluation with summary/export/dry-run modes
- **main()**: Routes to search/evaluate/reprocess/web based on flags

### 8.2 `src/config.py` — Configuration
- **Dataclasses**: SearchConfig, USAJobsConfig, AdzunaConfig, JoobleConfig, OutputConfig, DashboardConfig, WizardConfig, EvaluationConfig, AppConfig
- **WizardConfig**: Separate AI provider config for the setup wizard; falls back to EvaluationConfig if `provider` is empty
- **build_config()**: Merges YAML + CLI overrides; expands synonyms; loads `wizard:` and `evaluation:` sections
- **Environment variable fallback**: All API keys check env vars before config values

### 8.3 `src/aggregator.py` — Scraper Orchestrator
- **OUTPUT_COLUMNS**: Standard 15-column schema
- **extract_state()**: Regex-based US state extraction from location
- **normalize_jobspy_df()**: Maps JobSpy output to standard schema
- **apply_discipline_filter()**: Title-only include/exclude keyword filter
- **run_search()**: Parallel scraper execution with progressive save
- **reprocess()**: Re-filter/dedup from raw data without scraping

### 8.4 `src/dedup.py` — Deduplication Engine
- **3 layers**: URL exact match, fuzzy key (company|title|state), cross-source (title + company similarity)
- **normalize_url()**, **normalize_text()**, **normalize_company()**: Text normalization functions
- **companies_match()**: Substring matching + known job board detection
- **data_richness_score()**: Determines which duplicate to keep (+1000 for reviewed)
- **deduplicate()**: Main entry point; runs all 3 layers
- **Constants**: COMPANY_STOP_WORDS (40+), KNOWN_JOB_BOARDS (50+), STRIP_PARAMS

### 8.5 `src/exporter.py` — CSV/Excel Export
- **get_master_path()**: Returns path to master file
- **merge_and_export_csv()**: Core merge-on-save logic (load existing + concat + dedup + filter + save)
- **_export_excel()**: Excel output with auto-adjusted columns
- **migrate_old_files()**: One-time migration from timestamped files

### 8.6 `src/dashboard.py` — Streamlit Dashboard
- **Tab 1**: Job Listings with AG Grid, filters, review system, metrics
- **Tab 2**: Evaluation Results with color-coded scores, detail panel
- **Tab 3**: Setup — AI provider pickers (wizard + evaluation), setup wizard, search config, evaluator patterns, resume profile editor
- **Shared**: Review tracking (reviewed.json), run-search-from-dashboard

### 8.7 `src/scraper_jobspy.py` — Indeed/LinkedIn
- **scrape_jobs()**: Wraps python-jobspy; iterates search terms with delay

### 8.8 `src/scraper_usajobs.py` — USAJobs API
- **scrape_usajobs()**: REST API queries with Authorization-Key header

### 8.9 `src/scraper_adzuna.py` — Adzuna API
- **scrape_adzuna()**: Paginated REST API with rate limiting (2.5s delay)

### 8.10 `src/scraper_jooble.py` — Jooble API
- **scrape_jooble()**: Paginated REST API with JSON payload

### 8.11 `src/evaluator.py` — Evaluation Pipeline
- **prefilter_job()**: Stage 1 rule-based skip/rescue/boost
- **evaluate_single_job()**: Stage 2 async Claude API call with retry
- **evaluate_batch()**: Concurrent evaluation with backpressure and fail-fast
- **run_evaluation_pipeline()**: Full orchestration (prefilter → fetch descriptions → API score → persist)
- **estimate_cost()**: Dry-run cost estimation

### 8.12 `src/eval_persistence.py` — Evaluation Store
- **EvaluationStore**: JSON persistence class (load/save/is_evaluated/add/get/summary/export)

### 8.13 `src/description_fetcher.py` — Description Fetcher
- **fetch_description()**: Fetches HTML and extracts text via CSS selectors
- **fetch_missing_descriptions()**: Batch fetcher with rate limiting
- **Supports**: Greenhouse, Lever, Workday, iCIMS, LinkedIn, generic ATS

### 8.14 `src/resume_profile.py` — Resume Profile Loader
- **load_profile()**: Loads and validates resume_profile.json
- **profile_summary()**: Human-readable profile summary

---

## 9. Testing

### 9.1 Unit Tests (`tests/test_evaluator.py`)
- Tests for Stage 1 pre-filter patterns (skip, rescue, boost, pass)
- Validates known titles are correctly classified
- Run: `pytest tests/test_evaluator.py`

### 9.2 Integration Testing (Manual)

| Test | Command | Expected |
|------|---------|----------|
| Basic search | `python job_search.py --days 1` | Jobs saved to master CSV |
| Pre-filter only | `python job_search.py --eval-prefilter-only` | No API calls, shows skip/pass/boost counts |
| Cost estimate | `python job_search.py --eval-dry-run` | Shows job count + estimated cost |
| Full evaluation | `python job_search.py --evaluate-only --eval-days 1` | Jobs scored, evaluations.json updated |
| Dashboard | `python job_search.py --web` | Streamlit opens in browser |
| Reprocess | `python job_search.py --reprocess` | Re-filters from raw CSV |
| Export | `python job_search.py --eval-export out.csv --eval-min-score 60` | CSV with filtered evaluations |
| Summary | `python job_search.py --eval-summary` | Stats printed to terminal |

---

## 10. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| python-jobspy | latest | Indeed/LinkedIn scraping |
| pandas | latest | DataFrame manipulation |
| openpyxl | latest | Excel export |
| streamlit | latest | Dashboard web UI |
| streamlit-aggrid | latest | AG Grid component |
| pyyaml | latest | Config file loading |
| requests | latest | HTTP client (APIs, description fetching) |
| beautifulsoup4 | latest | HTML parsing (descriptions) |
| anthropic | latest | Anthropic (Claude) API client |
| openai | latest | OpenAI / Ollama API client |
| pdfplumber | latest | PDF resume text extraction |
| python-docx | latest | DOCX resume text extraction |

**Python**: 3.10+
**Platforms**: macOS, Linux, Windows

---

## 11. File Structure

```
pharma-job-search/
├── job_search.py                  # CLI entry point
├── pyproject.toml                 # Package config (pip installable)
├── config.yaml                    # User config (gitignored)
├── config.example.yaml            # Template config
├── requirements.txt               # Python dependencies
├── README.md                      # User documentation
├── CONTRIBUTING.md                # Contributor guide
├── CLAUDE.md                      # Architecture reference
├── LICENSE                        # MIT License
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py                  # Config loader + dataclasses (WizardConfig, EvaluationConfig)
│   ├── ai_client.py               # Multi-provider AI client (Anthropic, OpenAI, Ollama)
│   ├── setup_wizard.py            # AI-powered setup wizard (resume → full config)
│   ├── resume_parser.py           # Resume text extraction (PDF, DOCX, TXT)
│   ├── pattern_helpers.py         # Regex ↔ display string conversion
│   ├── aggregator.py              # Scraper orchestrator (ThreadPool)
│   ├── dedup.py                   # 3-layer deduplication
│   ├── exporter.py                # CSV/Excel merge-on-save
│   ├── dashboard.py               # Streamlit UI + AG Grid (3 tabs)
│   ├── scraper_jobspy.py          # Indeed/LinkedIn via JobSpy
│   ├── scraper_usajobs.py         # USAJobs REST API
│   ├── scraper_adzuna.py          # Adzuna REST API
│   ├── scraper_jooble.py          # Jooble REST API
│   ├── evaluator.py               # 2-stage AI evaluation pipeline
│   ├── eval_persistence.py        # Evaluation JSON store
│   ├── description_fetcher.py     # HTML description scraper
│   └── resume_profile.py          # Resume profile loader
├── tests/
│   ├── __init__.py
│   ├── test_evaluator.py          # Pre-filter unit tests
│   ├── test_ai_client.py          # Multi-provider AI client tests
│   ├── test_setup_wizard.py       # Setup wizard tests
│   ├── test_resume_parser.py      # Resume parser tests
│   └── test_pattern_helpers.py    # Pattern helper tests
├── assets/
│   ├── Job Listing.png            # Screenshot: Job Listings tab
│   ├── Job Evaluation.png         # Screenshot: Evaluation Results tab
│   └── Search Setup.png           # Screenshot: Setup tab
└── data/
    ├── resume_profile.example.json # Template profile
    ├── resume_profile.json         # User profile (gitignored)
    ├── evaluator_patterns.yaml     # Evaluator patterns (gitignored)
    ├── pharma_jobs.csv             # Master job data (generated)
    ├── pharma_jobs.xlsx            # Master Excel (generated)
    ├── pharma_jobs_raw.csv         # Raw pre-filter data (generated)
    ├── evaluations.json            # AI evaluations (generated)
    └── reviewed.json               # Review tracking (generated)
```

---

## 12. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Title-only discipline filtering | Prevents false positives from description boilerplate mentioning many disciplines |
| 3-layer dedup (URL → fuzzy → cross-source) | Catches duplicates that share no URLs but are the same job across boards |
| Reviewed jobs get +1000 richness boost | Ensures user-reviewed jobs survive dedup merges — never lose review state |
| Progressive save (each scraper saves immediately) | Partial results available if any scraper fails; faster feedback |
| 2-stage evaluation (rule-based → AI) | Pre-filter eliminates ~50% of jobs at zero API cost |
| Title-only jobs capped at score 50 (enforced in code) | AI prompt asks for the cap but doesn't reliably follow it; hard cap in `evaluate_single_job()` clamps score, recalculates bucket/rec, and prefixes domain_match with `[Title Only]` |
| Incremental evaluation saves (every 5 jobs) | Prevents data loss when browser refresh or interruption kills in-progress evaluation; `evaluate_batch()` passes store for incremental persist |
| Fail-fast after 3 auth errors | Stops wasting time on invalid API keys |
| Catch-up evaluation for missed jobs | Prevents jobs from permanently falling out of the eval window |
| Merge-on-save architecture | Master CSV is always current; no manual merge step |
| Synonym expansion in config | Reduces base search terms while maintaining coverage |
| Environment variable fallback for API keys | Supports CI/CD and keeps secrets out of config files |
| Single master file (no timestamps) | One source of truth; prevents file proliferation |
| Repost date tracking | Helps identify jobs being actively re-posted (signals urgency) |

---

## 13. Future Considerations

- Additional job board scrapers (Glassdoor, ZipRecruiter — currently blocked by anti-bot)
- Email/Slack notifications for high-score jobs
- Scheduled automated runs (cron/launchd)
- Historical analytics (score trends, application tracking)
- Multi-user support with separate profiles
- Application tracking (applied, interviewing, offered, rejected)
- Fine-tuned evaluation model for domain-specific scoring
