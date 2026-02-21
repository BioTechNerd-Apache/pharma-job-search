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

    # Ensure eval_status column exists in both DataFrames
    if existing is not None and "eval_status" not in existing.columns:
        existing["eval_status"] = ""
    if "eval_status" not in new_df.columns:
        new_df["eval_status"] = ""

    # Combine existing + new
    if existing is not None and not existing.empty:
        # Preserve eval_status from existing records: if a job already has
        # eval_status (evaluated/skipped), a fresh scrape of the same URL
        # should not overwrite it with "".
        existing_status = existing[["job_url", "eval_status"]].dropna(subset=["job_url"])
        existing_status = existing_status[existing_status["eval_status"].fillna("").astype(str).str.strip() != ""]
        status_map = dict(zip(existing_status["job_url"], existing_status["eval_status"]))

        combined = pd.concat([existing.dropna(axis=1, how="all"), new_df.dropna(axis=1, how="all")], ignore_index=True)
        logger.info(f"Combined existing ({len(existing)}) + new ({len(new_df)}) = {len(combined)} rows")
    else:
        combined = new_df.copy()
        status_map = {}

    # Deduplicate the combined data
    combined = deduplicate(combined)

    # Restore eval_status for jobs that had one before dedup
    if status_map and "eval_status" in combined.columns:
        for url, status in status_map.items():
            mask = combined["job_url"] == url
            if mask.any():
                combined.loc[mask, "eval_status"] = status

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

    # Ensure eval_status column exists (default "" = pending/unevaluated)
    if "eval_status" not in combined.columns:
        combined["eval_status"] = ""
    combined["eval_status"] = combined["eval_status"].fillna("")

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
