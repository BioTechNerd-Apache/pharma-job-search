#!/usr/bin/env python3
"""Pharma/Biotech Job Search Aggregator - CLI entry point."""

import argparse
import logging
import platform
import shutil
import stat
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
  python job_search.py --create-shortcut             # Create desktop shortcut

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

    # Setup wizard
    parser.add_argument("--setup", type=str, metavar="RESUME_FILE",
                        help="Run interactive setup wizard with a resume file (PDF, DOCX, or TXT)")

    # Shortcut
    parser.add_argument("--create-shortcut", action="store_true",
                        help="Create a desktop shortcut to launch the dashboard")

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


def _get_launch_command() -> str:
    """Return the command to launch the dashboard, depending on install method."""
    if shutil.which("pharma-job-search"):
        return "pharma-job-search --web"
    return f"{sys.executable} {Path(__file__).resolve()} --web"


def _create_macos_shortcut(launch_cmd: str) -> bool:
    """Create a .command shortcut on macOS Desktop."""
    shortcut = Path.home() / "Desktop" / "Pharma Job Search.command"
    shortcut.write_text(
        f"#!/bin/bash\n"
        f"# Pharma/Biotech Job Search Dashboard launcher\n"
        f"{launch_cmd} &\n"
        f"sleep 3\n"
        f"open http://localhost:8501\n"
    )
    shortcut.chmod(shortcut.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return True


def _create_windows_shortcut(launch_cmd: str) -> bool:
    """Create a .bat shortcut on Windows Desktop."""
    shortcut = Path.home() / "Desktop" / "Pharma Job Search.bat"
    shortcut.write_text(
        f"@echo off\r\n"
        f"REM Pharma/Biotech Job Search Dashboard launcher\r\n"
        f"start \"\" {launch_cmd}\r\n"
        f"timeout /t 5 /nobreak >nul\r\n"
        f"start http://localhost:8501\r\n"
    )
    return True


def _create_linux_shortcut(launch_cmd: str) -> bool:
    """Create a .desktop shortcut on Linux Desktop."""
    shortcut = Path.home() / "Desktop" / "pharma-job-search.desktop"
    shortcut.write_text(
        f"[Desktop Entry]\n"
        f"Type=Application\n"
        f"Name=Pharma Job Search\n"
        f"Comment=Launch the Pharma/Biotech Job Search Dashboard\n"
        f"Exec={launch_cmd}\n"
        f"Terminal=true\n"
        f"Categories=Utility;\n"
    )
    shortcut.chmod(shortcut.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return True


def create_shortcut() -> bool:
    """Create a platform-appropriate desktop shortcut to launch the dashboard."""
    launch_cmd = _get_launch_command()
    system = platform.system()

    try:
        if system == "Darwin":
            success = _create_macos_shortcut(launch_cmd)
        elif system == "Windows":
            success = _create_windows_shortcut(launch_cmd)
        elif system == "Linux":
            success = _create_linux_shortcut(launch_cmd)
        else:
            logger.error(f"Unsupported platform: {system}")
            return False
    except OSError as e:
        logger.error(f"Failed to create shortcut: {e}")
        return False

    if success:
        desktop = Path.home() / "Desktop"
        logger.info(f"Desktop shortcut created at {desktop}")
    return success


def cli_progress_callback(scraper_name, status, count=0):
    """Print-based progress callback for CLI use."""
    if status == "starting":
        logger.info(f"  Starting {scraper_name}...")
    elif status == "done":
        logger.info(f"  {scraper_name} done — {count} results")


from src.evaluator import load_jobs_csv, filter_jobs_by_time


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

    # --create-shortcut: create desktop shortcut (no config needed)
    if args.create_shortcut:
        success = create_shortcut()
        sys.exit(0 if success else 1)

    # --setup: run setup wizard before anything else
    if args.setup:
        from src.setup_wizard import run_cli_wizard
        success = run_cli_wizard(args.setup)
        sys.exit(0 if success else 1)

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
