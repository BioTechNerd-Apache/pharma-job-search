"""Adzuna API client: queries pharma/biotech jobs, normalizes to common schema."""

import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from .config import AdzunaConfig

logger = logging.getLogger(__name__)

ADZUNA_BASE_URL = "https://api.adzuna.com/v1/api/jobs/us/search"


def scrape_adzuna(config: AdzunaConfig, days: int = 7) -> pd.DataFrame:
    """Query Adzuna API for pharma/biotech roles. Returns empty DataFrame if not configured."""
    if not config.enabled:
        logger.info("Adzuna: skipping (no API credentials configured)")
        return pd.DataFrame()

    app_id = config.get_app_id()
    app_key = config.get_app_key()
    all_jobs = []

    for keyword in config.keywords:
        logger.info(f"Adzuna: searching for '{keyword}'...")

        for page in range(1, config.max_pages + 1):
            try:
                params = {
                    "app_id": app_id,
                    "app_key": app_key,
                    "what": keyword,
                    "where": "United States",
                    "results_per_page": config.results_per_page,
                    "sort_by": "date",
                    "max_days_old": days,
                }
                url = f"{ADZUNA_BASE_URL}/{page}"
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                results = data.get("results", [])
                if not results:
                    break

                for item in results:
                    location_str = item.get("location", {}).get("display_name", "")

                    salary_min = item.get("salary_min")
                    salary_max = item.get("salary_max")
                    try:
                        salary_min = float(salary_min) if salary_min else None
                        salary_max = float(salary_max) if salary_max else None
                    except (ValueError, TypeError):
                        salary_min = salary_max = None

                    date_posted = item.get("created")
                    try:
                        date_posted = datetime.strptime(
                            date_posted, "%Y-%m-%dT%H:%M:%SZ"
                        ).date()
                    except (ValueError, TypeError):
                        try:
                            date_posted = datetime.strptime(
                                date_posted[:10], "%Y-%m-%d"
                            ).date()
                        except (ValueError, TypeError):
                            date_posted = None

                    is_remote = False
                    if location_str:
                        is_remote = "remote" in location_str.lower()

                    contract_time = item.get("contract_time", "")
                    job_type = ""
                    if contract_time == "full_time":
                        job_type = "Full-time"
                    elif contract_time == "part_time":
                        job_type = "Part-time"
                    elif contract_time:
                        job_type = contract_time

                    all_jobs.append({
                        "title": item.get("title", ""),
                        "company": item.get("company", {}).get("display_name", ""),
                        "location": location_str,
                        "state": "",
                        "date_posted": date_posted,
                        "source": "adzuna",
                        "job_url": item.get("redirect_url", ""),
                        "salary_min": salary_min,
                        "salary_max": salary_max,
                        "is_remote": is_remote,
                        "job_type": job_type,
                        "description": item.get("description", ""),
                    })

                logger.info(
                    f"  Adzuna: page {page} returned {len(results)} results for '{keyword}'"
                )

                if len(results) < config.results_per_page:
                    break

                time.sleep(2.5)

            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    logger.warning(f"  Adzuna rate limited for '{keyword}', stopping pagination")
                    break
                logger.warning(f"  Adzuna error for '{keyword}' page {page}: {e}")
                break
            except Exception as e:
                logger.warning(f"  Adzuna error for '{keyword}' page {page}: {e}")
                break

        logger.info(f"  Adzuna: total {len(all_jobs)} results so far")

    if not all_jobs:
        return pd.DataFrame()

    return pd.DataFrame(all_jobs)
