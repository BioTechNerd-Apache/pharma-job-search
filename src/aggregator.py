"""Orchestrates scrapers, normalizes columns, merges, deduplicates, sorts."""

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .config import AppConfig, SearchConfig
from .scraper_jobspy import scrape_jobs
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


def _scrape_indeed(config):
    """Wrapper to run JobSpy scraper for Indeed only."""
    df = scrape_jobs(config.search, site="indeed")
    if not df.empty:
        df = normalize_jobspy_df(df)
    return ("Indeed", df)


def _scrape_linkedin(config):
    """Wrapper to run JobSpy scraper for LinkedIn only."""
    df = scrape_jobs(config.search, site="linkedin")
    if not df.empty:
        df = normalize_jobspy_df(df)
    return ("LinkedIn", df)


def _scrape_usajobs(config):
    """Wrapper to run USAJobs scraper and return (name, DataFrame)."""
    df = scrape_usajobs(config.usajobs, days=config.search.days)
    if not df.empty:
        for col in OUTPUT_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[OUTPUT_COLUMNS]
    return ("USAJobs", df)


def _scrape_adzuna(config):
    """Wrapper to run Adzuna scraper and return (name, DataFrame)."""
    df = scrape_adzuna(config.adzuna, days=config.search.days)
    if not df.empty:
        for col in OUTPUT_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df["state"] = df["location"].apply(extract_state)
        df = df[OUTPUT_COLUMNS]
    return ("Adzuna", df)


def _scrape_jooble(config):
    """Wrapper to run Jooble scraper and return (name, DataFrame)."""
    df = scrape_jooble(config.jooble, days=config.search.days)
    if not df.empty:
        for col in OUTPUT_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df["state"] = df["location"].apply(extract_state)
        df = df[OUTPUT_COLUMNS]
    return ("Jooble", df)


def _get_raw_path(config: AppConfig):
    """Return path to the raw intermediate CSV file."""
    from .config import PROJECT_ROOT
    raw_dir = PROJECT_ROOT / config.output.directory
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir / f"{config.output.filename_prefix}_raw.csv"


def run_search(config: AppConfig, progress_callback=None) -> pd.DataFrame:
    """Run all scrapers in parallel. Each scraper saves to the master CSV as soon as it finishes.

    Also saves raw normalized data (before filtering/dedup) to a raw intermediate file
    so that reprocessing can be done without re-scraping.

    Args:
        config: Application configuration.
        progress_callback: Optional callable(scraper_name, status, count=0).
            status is "starting" or "done".
    """
    from .exporter import merge_and_export_csv

    scraper_funcs = [_scrape_indeed, _scrape_linkedin, _scrape_usajobs, _scrape_adzuna, _scrape_jooble]
    scraper_names = ["Indeed", "LinkedIn", "USAJobs", "Adzuna", "Jooble"]
    csv_lock = threading.Lock()
    raw_lock = threading.Lock()
    total_saved = 0
    raw_path = _get_raw_path(config)

    # Clear old raw file at the start of a new search
    if raw_path.exists():
        raw_path.unlink()

    # Notify all scrapers starting
    if progress_callback:
        for name in scraper_names:
            progress_callback(name, "starting", count=0)

    logger.info("=== Starting all scrapers in parallel ===")

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_name = {}
        for func, name in zip(scraper_funcs, scraper_names):
            future = executor.submit(func, config)
            future_to_name[future] = name

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                scraper_name, df = future.result()
                count = len(df) if not df.empty else 0

                if not df.empty:
                    # Append raw normalized data to intermediate file (before filtering/dedup)
                    with raw_lock:
                        header = not raw_path.exists()
                        df.to_csv(raw_path, mode="a", index=False, header=header)
                        logger.info(f"{scraper_name}: appended {len(df)} raw rows to {raw_path.name}")

                    # Filter and save to master CSV (thread-safe)
                    filtered = apply_discipline_filter(df, config.search)
                    if not filtered.empty:
                        with csv_lock:
                            merge_and_export_csv(filtered, config)
                        total_saved += len(filtered)
                        logger.info(f"{scraper_name}: saved {len(filtered)} jobs to master CSV")

                logger.info(f"{scraper_name} returned {count} results")
                if progress_callback:
                    progress_callback(scraper_name, "done", count=count)
            except Exception as e:
                logger.error(f"{name} scraper failed: {e}")
                if progress_callback:
                    progress_callback(name, "done", count=0)

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
