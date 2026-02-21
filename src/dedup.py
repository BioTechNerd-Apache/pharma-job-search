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
