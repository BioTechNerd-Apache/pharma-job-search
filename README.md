# Pharma/Biotech Job Search Tool

A Python CLI + Streamlit dashboard that aggregates pharma/biotech job listings from 5 sources and uses AI to score them against your resume profile.

## Features

- **Multi-source aggregation**: Searches Indeed, LinkedIn (via [JobSpy](https://github.com/Bunsly/JobSpy)), USAJobs, Adzuna, and Jooble simultaneously
- **Smart deduplication**: 3-layer dedup (URL, fuzzy company+title+state, cross-source) eliminates duplicate listings
- **Discipline filtering**: Configurable include/exclude keyword filters keep results relevant to your field
- **AI-powered evaluation**: 2-stage pipeline — rule-based pre-filter skips obvious mismatches, then Claude Haiku scores remaining jobs against your resume profile
- **Interactive dashboard**: Streamlit UI with AG Grid for browsing, filtering, and reviewing jobs
- **Rolling data**: Merges new results into a master CSV/Excel file, preserving your review history
- **Repost detection**: Tracks when jobs are reposted across sources

## Quick Start

### 1. Clone and install

**macOS / Linux:**
```bash
git clone https://github.com/BioTechNerd-Apache/pharma-job-search.git
cd pharma-job-search
pip install -r requirements.txt
```

**Windows (Command Prompt or PowerShell):**
```cmd
git clone https://github.com/BioTechNerd-Apache/pharma-job-search.git
cd pharma-job-search
pip install -r requirements.txt
```

### 2. Configure

**macOS / Linux:**
```bash
cp config.example.yaml config.yaml
```

**Windows:**
```cmd
copy config.example.yaml config.yaml
```

Edit `config.yaml` to add your API keys (see [API Keys](#api-keys) below). The search terms, filters, and synonyms are pre-configured for pharma/biotech — customize them for your specific discipline.

### 3. Set up your resume profile (for AI evaluation)

**macOS / Linux:**
```bash
cp data/resume_profile.example.json data/resume_profile.json
```

**Windows:**
```cmd
copy data\resume_profile.example.json data\resume_profile.json
```

Edit `data/resume_profile.json` with your actual background. The AI evaluator uses this to score job fit.

### 4. Run a search

```bash
python job_search.py --days 1    # Search for jobs posted in the last 24 hours
```

### 5. View the dashboard

```bash
python job_search.py --web
```

Or double-click the launcher for your platform:
- **macOS**: `BioTechNerd-Apache's Job Search.command`
- **Windows**: `BioTechNerd-Apache's Job Search.bat`

> **Desktop shortcut:** Drag and drop the `.command` (Mac) or `.bat` (Windows) file from the project folder onto your Desktop for one-click dashboard access.

> **Dashboard search button:** The dashboard has a "Run New Search" button in the sidebar. Clicking it will scrape all 5 job boards for the last 7 days (the default in `config.yaml`). It does **not** run AI evaluation — that requires the CLI (`python job_search.py --evaluate`). To change the search window, edit the `days` value under `search:` in `config.yaml`.

## API Keys

| Source | Required? | Where to Register |
|--------|-----------|-------------------|
| Indeed | No | Works via JobSpy, no key needed |
| LinkedIn | No | Works via JobSpy, no key needed |
| USAJobs | Yes | [developer.usajobs.gov](https://developer.usajobs.gov/) |
| Adzuna | Yes | [developer.adzuna.com](https://developer.adzuna.com/) |
| Jooble | Yes | [jooble.org/api/about](https://jooble.org/api/about) |
| Anthropic | For AI eval | [console.anthropic.com](https://console.anthropic.com/settings/keys) |

You can set API keys in `config.yaml` or via environment variables:
- `USAJOBS_API_KEY`, `USAJOBS_EMAIL`
- `ADZUNA_APP_ID`, `ADZUNA_APP_KEY`
- `JOOBLE_API_KEY`
- `ANTHROPIC_API_KEY`

**Note**: Indeed and LinkedIn work without any API keys. You can start searching immediately after install.

## CLI Reference

### Search Commands

```bash
python job_search.py                    # Full search (default: last 7 days)
python job_search.py --days 1           # Last 24 hours
python job_search.py --days 14          # Last 2 weeks
python job_search.py --reprocess        # Re-filter/dedup from raw data (no scraping)
python job_search.py --web              # Launch dashboard only
```

### Evaluation Commands

```bash
python job_search.py --evaluate               # Search + evaluate new jobs
python job_search.py --evaluate-only           # Evaluate without searching
python job_search.py --eval-days 3            # Evaluate jobs from last 3 days
python job_search.py --eval-all               # Evaluate all unevaluated jobs
python job_search.py --eval-prefilter-only    # Rule-based filter only (no API cost)
python job_search.py --eval-dry-run           # Show count + cost estimate
python job_search.py --eval-summary           # Show evaluation statistics
python job_search.py --eval-export results.csv --eval-min-score 60  # Export results
python job_search.py --re-evaluate            # Force re-evaluation of scored jobs
```

## Dashboard

The Streamlit dashboard has two tabs:

- **Job Listings**: Browse all scraped jobs with sortable/filterable AG Grid. Mark jobs as reviewed.
- **Evaluation Results**: View AI-scored jobs with color-coded fit scores (green = strong fit, red = poor fit).

Both tabs share the review system — select rows and click "Mark Reviewed" to track which jobs you've looked at.

## Customization Guide

This tool ships pre-configured for **pharma/biotech scientist** roles. There are 4 layers to customize for your background:

### Layer 1: Search Terms (`config.yaml` → `search.terms` + `search.synonyms`)

**What they do**: These keywords are sent to job boards. Each term in `synonyms` auto-expands — e.g., searching for "cell gene therapy" also searches "CGT scientist", "CAR-T scientist", "gene therapy", "cell therapy".

**To customize**: Replace terms with your discipline's job titles and keywords. The synonym groups reduce the number of base terms you need.

### Layer 2: Discipline Filters (`config.yaml` → `search.filter_include` + `search.filter_exclude`)

**What they do**: After scraping, jobs are filtered by **title only** (not description — this prevents false positives). A job must match at least one `filter_include` keyword AND match zero `filter_exclude` keywords to be kept.

**To customize**: Replace include keywords with terms relevant to your field. The exclude list filters out irrelevant roles (sales, nursing, IT, etc.) — most of it is broadly useful, but review it for your domain.

### Layer 3: Pre-filter Patterns (`src/evaluator.py` → `SKIP_TITLE_PATTERNS`, `SKIP_DESCRIPTION_PATTERNS`, `RESCUE_PATTERNS`, `BOOST_PATTERNS`)

**What they do**: Before sending jobs to the AI for scoring (which costs money), a rule-based pre-filter runs:
- **Skip patterns** (~50 regex patterns on title, ~15 on description): Auto-skip obvious mismatches (e.g., VP roles, QC technicians, HPLC-focused positions, data scientists)
- **Rescue patterns** (~15 patterns): Override skips — if a job matches both skip AND rescue (e.g., "HPLC" in description but also "qPCR"), it is NOT skipped
- **Boost patterns** (~15 patterns): Jobs matching these get evaluated first (e.g., bioanalytical, gene therapy, CAR-T)

**To customize**: These are **hardcoded in `src/evaluator.py`** and are specific to pharma/biotech. You'll need to edit the regex pattern lists to match your field. Run `--eval-prefilter-only` to test your patterns without API cost.

### Layer 4: Resume Profile (`data/resume_profile.json`)

**What they do**: The AI evaluator (Claude Haiku) reads this profile to score how well each job matches your background. The profile includes your career history, technical platforms, regulatory experience, and fit domains.

**To customize**: Copy `data/resume_profile.example.json` to `data/resume_profile.json` and fill in your actual background. Key fields:

| Field | Purpose |
|-------|---------|
| `career_anchors` | Your work history — skills at each position |
| `core_technical_platforms` | Instruments/techniques you know |
| `regulatory_framework` | GLP, GMP, FDA experience etc. |
| `strongest_fit_domains` | Job types that match you best (scored highest) |
| `moderate_fit_domains` | Decent matches (scored moderately) |
| `skip_domains` | Poor matches (scored low) |
| `never_claim` | Skills/platforms you do NOT have (prevents false matches) |

### Getting Started Without API Keys

You can run a basic search with **zero API keys** — Indeed and LinkedIn work immediately via JobSpy:

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml
python job_search.py --days 1          # Scrapes Indeed + LinkedIn
python job_search.py --web             # View results in dashboard
```

Add USAJobs/Adzuna/Jooble keys for more sources. Add an Anthropic key to enable AI evaluation.

## Project Structure

```
job_search.py                # CLI entry point
config.yaml                  # Your configuration (not tracked by git)
config.example.yaml          # Template configuration
BioTechNerd-Apache's Job Search.command  # macOS dashboard launcher (double-click)
BioTechNerd-Apache's Job Search.bat     # Windows dashboard launcher (double-click)
src/
  config.py                  # YAML loader + dataclasses
  aggregator.py              # Orchestrator with parallel scraping
  dedup.py                   # 3-layer deduplication
  exporter.py                # CSV/Excel merge-on-save
  dashboard.py               # Streamlit UI with AG Grid
  scraper_jobspy.py          # Indeed/LinkedIn via python-jobspy
  scraper_usajobs.py         # USAJobs REST API
  scraper_adzuna.py          # Adzuna REST API
  scraper_jooble.py          # Jooble REST API
  evaluator.py               # 2-stage evaluation pipeline
  eval_persistence.py        # JSON persistence for evaluations
  description_fetcher.py     # HTML scraping for job descriptions
  resume_profile.py          # Resume profile loader
tests/
  test_evaluator.py          # Tests for rule-based pre-filter
data/
  resume_profile.json        # Your resume profile (not tracked)
  resume_profile.example.json # Template resume profile
  pharma_jobs.csv            # Master job data (generated)
  pharma_jobs.xlsx           # Master job data (generated)
```

## Requirements

- Python 3.10+
- macOS, Linux, or Windows
- See `requirements.txt` for Python dependencies

## License

MIT License. See [LICENSE](LICENSE) for details.
