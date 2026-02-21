"""Jooble API client: queries pharma/biotech jobs, normalizes to common schema."""

import logging
import re
import time
from datetime import datetime

import pandas as pd
import requests

from .config import JoobleConfig

logger = logging.getLogger(__name__)

JOOBLE_BASE_URL = "https://jooble.org/api"


def _parse_salary(salary_str: str):
    """Parse Jooble salary string (e.g., '$80,000 - $120,000') into min/max floats."""
    if not salary_str or not isinstance(salary_str, str):
        return None, None
    numbers = re.findall(r"[\d]+\.?\d*", salary_str.replace(",", ""))
    try:
        if len(numbers) >= 2:
            return float(numbers[0]), float(numbers[1])
        elif len(numbers) == 1:
            return float(numbers[0]), None
    except (ValueError, TypeError):
        pass
    return None, None


def scrape_jooble(config: JoobleConfig, days: int = 7) -> pd.DataFrame:
    """Query Jooble API for pharma/biotech roles. Returns empty DataFrame if not configured."""
    if not config.enabled:
        logger.info("Jooble: skipping (no API key configured)")
        return pd.DataFrame()

    api_key = config.get_api_key()
    url = f"{JOOBLE_BASE_URL}/{api_key}"
    all_jobs = []

    for keyword in config.keywords:
        logger.info(f"Jooble: searching for '{keyword}'...")

        for page in range(1, config.max_pages + 1):
            try:
                payload = {
                    "keywords": keyword,
                    "location": "United States",
                    "page": str(page),
                }
                resp = requests.post(url, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                jobs = data.get("jobs", [])
                if not jobs:
                    break

                for item in jobs:
                    location_str = item.get("location", "")

                    salary_min, salary_max = _parse_salary(item.get("salary", ""))

                    date_str = item.get("updated", "")
                    date_posted = None
                    if date_str:
                        try:
                            date_posted = datetime.strptime(
                                date_str[:10], "%Y-%m-%d"
                            ).date()
                        except (ValueError, TypeError):
                            date_posted = None

                    is_remote = False
                    if location_str:
                        is_remote = "remote" in location_str.lower()
                    job_type_str = item.get("type", "")
                    if job_type_str and "remote" in job_type_str.lower():
                        is_remote = True

                    all_jobs.append({
                        "title": item.get("title", ""),
                        "company": item.get("company", ""),
                        "location": location_str,
                        "state": "",
                        "date_posted": date_posted,
                        "source": "jooble",
                        "job_url": item.get("link", ""),
                        "salary_min": salary_min,
                        "salary_max": salary_max,
                        "is_remote": is_remote,
                        "job_type": job_type_str if job_type_str else "",
                        "description": item.get("snippet", ""),
                    })

                logger.info(
                    f"  Jooble: page {page} returned {len(jobs)} results for '{keyword}'"
                )

                total_count = data.get("totalCount", 0)
                if len(all_jobs) >= total_count:
                    break

                time.sleep(2)

            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    logger.warning(
                        f"  Jooble rate limited for '{keyword}', stopping pagination"
                    )
                    break
                logger.warning(f"  Jooble error for '{keyword}' page {page}: {e}")
                break
            except Exception as e:
                logger.warning(f"  Jooble error for '{keyword}' page {page}: {e}")
                break

        logger.info(f"  Jooble: total {len(all_jobs)} results so far")

    if not all_jobs:
        return pd.DataFrame()

    return pd.DataFrame(all_jobs)
