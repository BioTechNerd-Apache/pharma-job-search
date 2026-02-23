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
