# Pharma/Biotech Job Search — Detailed Architecture

## System Overview

```
+===========================================================================+
|                           job_search.py (CLI)                              |
|                                                                            |
|  --setup resume.pdf    --days 7    --evaluate-only    --web                |
|  Setup Wizard          Search      Evaluation         Dashboard            |
+===========+============+============+=============+===========+============+
            |            |            |             |
            v            v            v             v
     setup_wizard.py  aggregator.py  evaluator.py  dashboard.py
            |            |            |             |
            v            v            v             v
     config.yaml      pharma_jobs   evaluations   Streamlit UI
     resume_profile   .csv/.xlsx    .json         (3 tabs)
     evaluator_       reviewed.json
     patterns.yaml
```

---

## 1. Setup Wizard Pipeline

```
resume.pdf / .docx / .txt
        |
        v
+-------------------+
| resume_parser.py  |  Extract text (pdfplumber / python-docx)
|                   |  Max 15,000 chars, min 50 chars validation
+--------+----------+
         |
         v
+-------------------+
| setup_wizard.py   |  3 sequential AI calls via ai_client.py:
|                   |
|  Call 1: Resume   |  -> data/resume_profile.json
|    Profile JSON   |     {career_anchors, core_technical_platforms,
|                   |      regulatory_framework, strongest_fit_domains,
|                   |      moderate_fit_domains, never_claim}
|                   |
|  Call 2: Search   |  -> config.yaml (search section)
|    Config         |     {terms, synonyms, filter_include, filter_exclude,
|                   |      priority_terms}
|                   |
|  Call 3: Eval     |  -> data/evaluator_patterns.yaml
|    Patterns       |     {skip_title_patterns, skip_description_patterns,
|                   |      rescue_patterns, boost_patterns}
+--------+----------+
         |
         v
  Preview/Approval Step -> User confirms before saving
  Backs up existing files (.mybackup) before overwrite
```

---

## 2. Job Search & Aggregation Pipeline

```
config.yaml
  |
  |  search.terms (~37 terms)
  |  search.sites (indeed, linkedin)
  |  search.priority_terms (~10 terms, 200 results each)
  |  search.synonyms (term expansion)
  |  API configs (usajobs, adzuna, jooble)
  |
  v
+====================================================================+
|                      aggregator.py                                  |
|                                                                     |
|  ThreadPoolExecutor (5 workers) + shared work queue                 |
|  Per-site Semaphore(2) — max 2 concurrent per site                 |
|  5-second delay between consecutive same-site calls                 |
|                                                                     |
|  EXECUTION ORDER:                                                   |
|  1. API scrapers run first (single task each, ~2 min total):        |
|     +------------------+  +----------------+  +----------------+    |
|     | scraper_usajobs  |  | scraper_adzuna |  | scraper_jooble |    |
|     | REST API         |  | REST API       |  | REST API       |    |
|     | Federal jobs     |  | Paginated      |  | Returns snippet|    |
|     | Email+Key auth   |  | App ID+Key     |  | only (not full |    |
|     +------------------+  +----------------+  | description)   |    |
|                                               +----------------+    |
|                                                                     |
|  2. Indeed/LinkedIn pairs interleaved for balanced load:             |
|     +------------------+                                            |
|     | scraper_jobspy   |  Each (site, term) = 1 work queue item     |
|     | python-jobspy    |  Priority terms get 200 results            |
|     | Indeed+LinkedIn  |  Regular terms get 100 results             |
|     +------------------+                                            |
|                                                                     |
|  NORMALIZE: All scrapers -> 16-column standard schema               |
|  title, company, location, state, date_posted, reposted_date,      |
|  days_since_posted, source, job_url, job_url_direct, salary_min,    |
|  salary_max, is_remote, job_type, description, eval_status          |
|                                                                     |
|  DISCIPLINE FILTER (on title only):                                 |
|  - filter_include (~60 keywords): biolog, bioanalyt, molecular...   |
|  - filter_exclude (~100 keywords): food scientist, data scientist...|
|  - Must match at least 1 include AND 0 excludes                     |
|                                                                     |
|  PROGRESSIVE SAVE: Each completed (site,term) saves immediately     |
|  Thread-safe via lock                                               |
+============================+=======================================+
                             |
                             v
+====================================================================+
|                         dedup.py                                    |
|                                                                     |
|  3-LAYER DEDUPLICATION:                                             |
|                                                                     |
|  Layer 1: EXACT URL MATCH                                           |
|    Strip tracking params: utm_*, fbclid, gclid, ref, refId,        |
|    trackingId, trk, tk                                              |
|                                                                     |
|  Layer 2: FUZZY MATCH                                               |
|    Key = normalized(company + "|" + title + "|" + state)            |
|    Company normalization: lowercase, strip 40+ stop words           |
|    (inc, corp, llc, company, solutions, group, services, etc.)      |
|                                                                     |
|  Layer 3: CROSS-SOURCE MATCH                                        |
|    Same title + similar company across different sources             |
|    Recognizes 50+ known job board domains                           |
|                                                                     |
|  SURVIVOR SELECTION (data_richness_score):                          |
|    +1 per non-empty field (description length, salary, direct URL)  |
|    +1000 for reviewed jobs (never lose user work)                   |
|    Highest score survives, others discarded                         |
|                                                                     |
|  REPOST TRACKING:                                                   |
|    Extracts all date_posted values from duplicate groups             |
|    Stores earliest as reposted_date on survivor                     |
|                                                                     |
|  FUZZY-KEY HELPERS (for cross-run repost blocking):                 |
|    make_fuzzy_key(title, company, state)                            |
|      -> canonical key matching Layer 2 format                       |
|    load_reviewed_fkeys() -> set of historically reviewed fkeys      |
|    load_reviewed_fkeys_raw() -> {fkey: {url, timestamp}}            |
|    save_reviewed_fkeys(dict) -> writes data/reviewed_fkeys.json     |
+============================+=======================================+
                             |
                             v
+====================================================================+
|                        exporter.py                                  |
|                                                                     |
|  MERGE-ON-SAVE to single master file (no timestamped exports):      |
|  - data/pharma_jobs.csv   (master CSV)                              |
|  - data/pharma_jobs.xlsx  (master Excel, auto-adjusted columns)     |
|  - data/pharma_jobs_raw.csv (pre-filter/dedup raw, for reprocess)   |
|                                                                     |
|  Preserves eval_status column across merges                         |
|  migrate_old_files(): one-time merge of legacy timestamped CSVs     |
|                                                                     |
|  FUZZY-KEY FILTER (cross-run repost blocking):                      |
|    Runs before deduplicate() on every merge:                        |
|    1. Load reviewed_fkeys.json (company|title|state keys)           |
|    2. Drop any new row whose fkey matches a reviewed fkey           |
|       UNLESS its job_url is the canonical reviewed URL              |
|    -> Prevents reposted jobs (new LinkedIn ID / Indeed jk= param)   |
|       from re-entering master CSV after being reviewed              |
|                                                                     |
|  FIRST-RUN MIGRATION:                                               |
|    On first call after upgrade, if reviewed_fkeys.json is absent:   |
|    Iterates master CSV rows that are in reviewed.json               |
|    Builds fkeys retroactively (~3,500-4,000 entries)                |
|    Writes reviewed_fkeys.json — migration never runs again          |
+====================================================================+
```

---

## 3. Evaluation Pipeline (Detailed)

```
pharma_jobs.csv + evaluations.json + reviewed.json
        |
        v
+========================================================================+
|                    evaluator.py — run_evaluation_pipeline()             |
|                                                                         |
|  STAGE 0: REVIEWED-JOB SKIP (new)                                      |
|  +------------------------------------------------------------------+  |
|  | Load data/reviewed.json                                          |  |
|  | For each job in pipeline:                                        |  |
|  |   IF reviewed_at exists AND not in evaluations.json:             |  |
|  |     -> Stamp as "manually_reviewed" (score=0) in evaluations.json|  |
|  |     -> Update eval_status="skipped" in master CSV                |  |
|  |     -> Remove from pipeline (don't waste API calls)              |  |
|  |                                                                  |  |
|  | PURPOSE: Jobs reviewed in Jobs tab before eval pipeline ran      |  |
|  |          should never be sent to AI for scoring                  |  |
|  +------------------------------------------------------------------+  |
|        |                                                                |
|        v                                                                |
|  STAGE 1: RULE-BASED PRE-FILTER (zero API cost)                        |
|  +------------------------------------------------------------------+  |
|  | Load patterns from data/evaluator_patterns.yaml                  |  |
|  | (falls back to built-in defaults if file missing)                |  |
|  |                                                                  |  |
|  | For each job, check title + description against:                 |  |
|  |                                                                  |  |
|  | SKIP PATTERNS (~50 title, ~15 description):                      |  |
|  |   VP, QC analyst, HPLC scientist, data scientist, postdoc,      |  |
|  |   process development scientist, bioinformatics director, etc.   |  |
|  |                                                                  |  |
|  | RESCUE PATTERNS (~15, override skip):                            |  |
|  |   qPCR, flow cytometry, gene therapy, AAV, CAR-T, GLP,          |  |
|  |   bioanalytical, cell therapy, CRISPR, etc.                      |  |
|  |   -> If both skip AND rescue match, job is EVALUATED (not skip)  |  |
|  |                                                                  |  |
|  | BOOST PATTERNS (~15, priority evaluation):                       |  |
|  |   bioanalytical, gene therapy, CAR-T, cell therapy, etc.         |  |
|  |   -> Boosted jobs evaluated first in Stage 2                     |  |
|  |                                                                  |  |
|  | Result per job: action=skip|evaluate|boost + reason + pattern    |  |
|  | Skipped -> score=0, fit_bucket="prefilter_skip" in evaluations   |  |
|  +------------------------------------------------------------------+  |
|        |                                                                |
|        v                                                                |
|  ALREADY-EVALUATED CHECK                                                |
|  +------------------------------------------------------------------+  |
|  | For remaining jobs, check evaluations.json by job_url            |  |
|  | IF already evaluated AND --re-evaluate not set -> SKIP           |  |
|  | This prevents re-scoring and saves API cost                      |  |
|  +------------------------------------------------------------------+  |
|        |                                                                |
|        v                                                                |
|  STAGE 1.5: DESCRIPTION ENRICHMENT                                      |
|  +------------------------------------------------------------------+  |
|  | description_fetcher.py                                           |  |
|  |                                                                  |  |
|  | For jobs with missing/thin descriptions:                         |  |
|  |   _is_substantive_description() checks:                         |  |
|  |     - Length >= 200 chars                                        |  |
|  |     - Contains keywords: qualifications, requirements,           |  |
|  |       responsibilities, experience, skills, duties,              |  |
|  |       bachelor, master, phd, degree, years of experience         |  |
|  |                                                                  |  |
|  | IF not substantive -> fetch from job URL:                        |  |
|  |   1. Try job_url_direct first (employer ATS page)                |  |
|  |   2. Fall back to job_url (job board page)                       |  |
|  |   3. CSS selectors for: Greenhouse, Lever, Workday, iCIMS,      |  |
|  |      LinkedIn, generic patterns (class/id with "job-desc")      |  |
|  |   4. Rate limit: 1.5s delay between requests                    |  |
|  |   5. Timeout: 15s per URL                                       |  |
|  |   6. Min 200 chars, max 10,000 chars                            |  |
|  |                                                                  |  |
|  | NOTE: Jooble API only returns snippets, and jooble.org blocks    |  |
|  |       scraping (403). Many Jooble jobs stay title-only.          |  |
|  | NOTE: Expired postings (404/redirect) also stay title-only.      |  |
|  +------------------------------------------------------------------+  |
|        |                                                                |
|        v                                                                |
|  STAGE 2: AI SCORING                                                    |
|  +------------------------------------------------------------------+  |
|  | ai_client.py (Anthropic / OpenAI / Ollama)                       |  |
|  |                                                                  |  |
|  | SYSTEM PROMPT: Built from data/resume_profile.json               |  |
|  |   Includes: career anchors, technical platforms, regulatory      |  |
|  |   framework, strongest/moderate fit domains, never_claim list    |  |
|  |                                                                  |  |
|  | USER PROMPT (two variants):                                      |  |
|  |   IF substantive description available:                          |  |
|  |     "Evaluate this job for candidate fit:"                       |  |
|  |     + title, company, full description                           |  |
|  |     "Score based on overlap between candidate skills and         |  |
|  |      STATED job requirements."                                   |  |
|  |                                                                  |  |
|  |   IF title-only (no substantive description):                    |  |
|  |     "Evaluate this job for candidate fit                         |  |
|  |      (TITLE ONLY — cap score at 50 max):"                       |  |
|  |     + title, company                                             |  |
|  |     "Score based ONLY on what the title explicitly indicates.    |  |
|  |      Do NOT infer specific techniques."                          |  |
|  |                                                                  |  |
|  | AI RESPONSE (JSON):                                              |  |
|  |   {fit_score: 0-100, fit_bucket: strong|moderate|weak|poor,      |  |
|  |    recommendation: apply|maybe|skip,                             |  |
|  |    domain_match: "Bioanalytical",                                |  |
|  |    reasoning: "explanation text",                                |  |
|  |    matching_skills: [...], missing_skills: [...]}                 |  |
|  |                                                                  |  |
|  | POST-PROCESSING:                                                 |  |
|  |   IF title-only (description_available=False):                   |  |
|  |     - Hard cap: fit_score = min(fit_score, 50)                   |  |
|  |     - Recalculate bucket/recommendation from capped score        |  |
|  |     - Prefix domain_match with "[Title Only] "                   |  |
|  |                                                                  |  |
|  | RETRY LOGIC:                                                     |  |
|  |   Exponential backoff: 5s, 10s, 20s, 40s + random jitter        |  |
|  |   Max retries: 5 (configurable)                                  |  |
|  |   3 consecutive auth failures = abort entire batch               |  |
|  |                                                                  |  |
|  | INCREMENTAL SAVE:                                                |  |
|  |   Every 5 completed evaluations -> save evaluations.json         |  |
|  |   Prevents data loss if interrupted mid-batch                    |  |
|  +------------------------------------------------------------------+  |
|        |                                                                |
|        v                                                                |
|  PERSISTENCE                                                            |
|  +------------------------------------------------------------------+  |
|  | eval_persistence.py — EvaluationStore                            |  |
|  |                                                                  |  |
|  | evaluations.json structure:                                      |  |
|  | {                                                                |  |
|  |   "metadata": {                                                  |  |
|  |     "last_evaluation_run": "ISO timestamp",                      |  |
|  |     "total_evaluated": count,                                    |  |
|  |     "model_used": "claude-haiku-4-5-20251001",                   |  |
|  |     "provider": "anthropic"                                      |  |
|  |   },                                                             |  |
|  |   "evaluations": {                                               |  |
|  |     "job_url": {                                                 |  |
|  |       "fit_score": 0-100,                                        |  |
|  |       "fit_bucket": "strong|moderate|weak|poor|                  |  |
|  |                      prefilter_skip|manually_reviewed|error",     |  |
|  |       "recommendation": "apply|maybe|skip",                      |  |
|  |       "domain_match": "Bioanalytical",                           |  |
|  |       "reasoning": "...",                                        |  |
|  |       "description_available": true|false,                       |  |
|  |       "evaluated_timestamp": "ISO timestamp",                    |  |
|  |       "input_tokens": N,                                         |  |
|  |       "output_tokens": N                                         |  |
|  |     }                                                            |  |
|  |   }                                                              |  |
|  | }                                                                |  |
|  |                                                                  |  |
|  | SPECIAL FIT_BUCKET VALUES:                                       |  |
|  |   "prefilter_skip" = Stage 1 rule-based skip (score=0)           |  |
|  |   "manually_reviewed" = Stage 0 reviewed-job skip (score=0)      |  |
|  |   "error" = AI call failed after all retries                     |  |
|  |                                                                  |  |
|  | CSV WRITEBACK:                                                   |  |
|  |   eval_status column in pharma_jobs.csv updated to               |  |
|  |   "evaluated" or "skipped" for processed jobs                    |  |
|  +------------------------------------------------------------------+  |
+========================================================================+
```

---

## 4. Dashboard (Streamlit)

```
+========================================================================+
|                         dashboard.py                                    |
|                         localhost:8501                                   |
|                                                                         |
|  +------------------+--------------------+-------------------+          |
|  | Tab 1:           | Tab 2:             | Tab 3:            |          |
|  | Job Listings     | Evaluation Results | Setup / Profile   |          |
|  +------------------+--------------------+-------------------+          |
|                                                                         |
|  TAB 1: JOB LISTINGS (Safety net for jobs that missed eval pipeline)    |
|  +------------------------------------------------------------------+  |
|  | SIDEBAR FILTERS:                                                  |  |
|  |   [x] Unevaluated only (N remaining)  <- eval_status="" in CSV   |  |
|  |   [x] Unreviewed only (N remaining)   <- no reviewed_at          |  |
|  |   Job Board: [indeed] [jooble] [linkedin]                        |  |
|  |   State: [dropdown]                                              |  |
|  |   Remote: All / Remote Only / On-site Only                       |  |
|  |   [ ] Only show jobs with salary info                            |  |
|  |   [ ] Only show reposted jobs                                    |  |
|  |                                                                  |  |
|  | METRICS: | Unreviewed | Sources | States | Reposted |            |  |
|  |                                                                  |  |
|  | AG Grid: job_code, title, company, location, state, posted,      |  |
|  |   reposted_date, days_old, source, job_url, job_url_direct,      |  |
|  |   salary, remote, job_type, reviewed_at                          |  |
|  |                                                                  |  |
|  | ACTIONS: [Mark Reviewed] [Mark Unreviewed]                       |  |
|  | DETAIL PANEL: Selected row details + [Copy Job Info] expander    |  |
|  +------------------------------------------------------------------+  |
|                                                                         |
|  TAB 2: EVALUATION RESULTS (Daily review queue)                         |
|  +------------------------------------------------------------------+  |
|  | SIDEBAR FILTERS:                                                  |  |
|  |   Min Fit Score: [0-----slider-----100]                          |  |
|  |   Recommendation: [apply] [maybe] [skip]                         |  |
|  |   Fit Bucket: [strong] [moderate] [weak] [poor] [prefilter_skip] |  |
|  |               [manually_reviewed] [error]                        |  |
|  |   [x] Unreviewed only (N remaining)                              |  |
|  |   [ ] Title-only jobs only                                       |  |
|  |   Source, State, Remote filters                                  |  |
|  |                                                                  |  |
|  | METRICS: | Unreviewed | Apply | Maybe | Avg Score |              |  |
|  |                                                                  |  |
|  | AG Grid: fit_score (color-coded), fit_bucket, recommendation,    |  |
|  |   description_available (Full/Title Only), title, company,       |  |
|  |   domain_match, location, state, posted, days_old, source,       |  |
|  |   job_url, job_url_direct, evaluated_timestamp, reviewed_at      |  |
|  |                                                                  |  |
|  | FIT SCORE COLOR CODING:                                          |  |
|  |   >= 70: green     (strong)                                      |  |
|  |   >= 55: blue      (moderate)                                    |  |
|  |   >= 40: orange    (weak)                                        |  |
|  |   < 40:  red       (poor)                                        |  |
|  |                                                                  |  |
|  | ACTIONS: [Mark Reviewed] [Mark Unreviewed]                       |  |
|  | DETAIL PANEL: Score, reasoning, links + [Copy Job Info] expander |  |
|  +------------------------------------------------------------------+  |
|                                                                         |
|  TAB 3: SETUP / PROFILE                                                 |
|  +------------------------------------------------------------------+  |
|  | Config editor (YAML display)                                     |  |
|  | Resume profile viewer (JSON display)                             |  |
|  | Setup wizard launcher (with resume upload)                       |  |
|  +------------------------------------------------------------------+  |
|                                                                         |
|  REVIEW DATA FLOW:                                                      |
|  +------------------------------------------------------------------+  |
|  | User clicks checkbox -> selectionChanged -> stored in             |  |
|  |   st.session_state (keyed by job_url)                            |  |
|  | User clicks [Mark Reviewed]:                                      |  |
|  |   -> writes {job_url: "YYYY-MM-DD HH:MM"} to reviewed.json      |  |
|  |   -> writes {company|title|state: {url, timestamp}}              |  |
|  |      to reviewed_fkeys.json (blocks future reposts)              |  |
|  | User clicks [Undo Review]:                                        |  |
|  |   -> removes entry from reviewed.json                            |  |
|  |   -> removes matching fkey from reviewed_fkeys.json              |  |
|  | getRowId(job_url) ensures stable row identity across reruns      |  |
|  | Deterministic sort (fit_score desc + job_url asc) prevents       |  |
|  |   row rearrangement on selectionChanged reruns                   |  |
|  +------------------------------------------------------------------+  |
+========================================================================+
```

---

## 5. Data Flow Summary

```
                    DAILY WORKFLOW
                    =============

    MORNING: Scrape new jobs
    +-----------+
    | Search    |  python job_search.py --days 1
    | Pipeline  |  -> Scrape 5 sources -> Filter -> Dedup -> Save CSV
    +-----------+
         |
         v
    MORNING: Evaluate new jobs
    +-----------+
    | Eval      |  python job_search.py --evaluate-only --eval-days 1
    | Pipeline  |  -> Stage 0 (skip reviewed) -> Stage 1 (prefilter)
    +-----------+     -> Stage 1.5 (fetch descriptions) -> Stage 2 (AI score)
         |
         v
    DURING DAY: Review in dashboard
    +-----------+
    | Dashboard |  python job_search.py --web
    | Tab 2     |  -> Review evaluated jobs -> Mark reviewed
    +-----------+  -> "Unreviewed" count goes to 0
         |
         v
    SAFETY NET: Check Jobs tab
    +-----------+
    | Dashboard |  -> Any unevaluated+unreviewed jobs?
    | Tab 1     |  -> Review manually or run eval to process them
    +-----------+  -> "Unreviewed" count goes to 0

    END OF DAY: Both tabs show 0 unreviewed = all clear
```

---

## 6. Key Design Decisions & Logic Details

### Scoring Logic
| Condition | Score Range | Bucket | Recommendation |
|-----------|------------|--------|----------------|
| Full description, strong match | 70-100 | strong | apply |
| Full description, moderate match | 55-69 | moderate | maybe |
| Full description, weak match | 40-54 | weak | skip |
| Full description, poor match | 0-39 | poor | skip |
| **Title-only: hard cap** | **0-50 max** | weak/poor | maybe/skip |
| Pre-filter skip | 0 | prefilter_skip | skip |
| Manually reviewed | 0 | manually_reviewed | skip |
| API error | 0 | error | skip |

### description_available vs Title-Only Evaluation
- `description_available` = True if description field is non-empty and not "nan"
- `_is_substantive_description()` = True if 200+ chars AND contains job keywords
- **These can disagree**: A job can have `description_available=True` (shows "Full" in Info column) but still be evaluated as title-only if the description is generic boilerplate without real requirements
- Title-only evaluated jobs get `domain_match` prefixed with "[Title Only]"

### Re-evaluation Behavior
- `--re-evaluate` flag overwrites existing scores completely
- Without the flag, already-evaluated jobs are skipped (keyed by job_url)
- No score averaging or merging — full replacement only

### Reviewed Job Protection
- reviewed.json tracks all review timestamps
- Dedup gives reviewed jobs +1000 richness bonus (never lose user work)
- Stage 0 in eval pipeline stamps reviewed-but-unevaluated jobs as manually_reviewed
- eval_status in CSV updated so Jobs tab filter stays in sync
- reviewed_fkeys.json indexes company|title|state for every reviewed job
  so reposted listings (new LinkedIn ID / new Indeed jk= param) are dropped
  before entering master CSV — prevents reviewed jobs from resurfacing

### Jooble API Limitation
- Jooble API returns only a `snippet` (short preview), not full job description
- jooble.org blocks scraping (HTTP 403)
- `job_url_direct` is typically empty for Jooble jobs
- Result: Many Jooble jobs can only be evaluated as title-only (capped at 50)

### AI Provider Abstraction
```
ai_client.py
  |
  +-- AnthropicClient (Claude Haiku, Sonnet, Opus)
  |     API key: config.evaluation.api_key or ANTHROPIC_API_KEY env
  |
  +-- OpenAIClient (GPT-4o, GPT-4o-mini, etc.)
  |     API key: config.evaluation.api_key or OPENAI_API_KEY env
  |     Custom base_url for Together AI, Fireworks, Groq, etc.
  |
  +-- OllamaClient (local models via localhost:11434)
        No API key needed
        Requires sufficient RAM for model
```

### Thread Safety
- Progressive save uses threading lock for CSV writes
- Per-site Semaphore(2) prevents overwhelming job boards
- 5-second configurable delay between same-site calls
- Incremental eval saves every 5 jobs (atomic write to evaluations.json)
