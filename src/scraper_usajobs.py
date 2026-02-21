"""USAJobs API client: queries federal pharma/biotech jobs, normalizes to common schema."""

import logging
from datetime import datetime, timedelta

import pandas as pd
import requests

from .config import USAJobsConfig

logger = logging.getLogger(__name__)

USAJOBS_BASE_URL = "https://data.usajobs.gov/api/search"


def scrape_usajobs(config: USAJobsConfig, days: int = 7) -> pd.DataFrame:
    """Query USAJobs API for pharma/biotech roles. Returns empty DataFrame if not configured."""
    if not config.enabled:
        logger.info("USAJobs: skipping (no API key configured)")
        return pd.DataFrame()

    api_key = config.get_api_key()
    email = config.get_email()

    headers = {
        "Authorization-Key": api_key,
        "User-Agent": email,
        "Host": "data.usajobs.gov",
    }

    date_from = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    all_jobs = []

    for keyword in config.keywords:
        logger.info(f"USAJobs: searching for '{keyword}'...")
        try:
            params = {
                "Keyword": keyword,
                "DatePosted": days,
                "ResultsPerPage": 100,
            }
            resp = requests.get(USAJOBS_BASE_URL, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("SearchResult", {}).get("SearchResultItems", [])
            for item in results:
                pos = item.get("MatchedObjectDescriptor", {})
                locations = pos.get("PositionLocation", [])
                location_str = locations[0].get("LocationName", "") if locations else ""

                # Extract state from location
                state = ""
                if ", " in location_str:
                    parts = location_str.split(", ")
                    state = parts[-1].strip() if len(parts) >= 2 else ""

                salary_min = None
                salary_max = None
                remuneration = pos.get("PositionRemuneration", [])
                if remuneration:
                    salary_min = remuneration[0].get("MinimumRange")
                    salary_max = remuneration[0].get("MaximumRange")
                    try:
                        salary_min = float(salary_min) if salary_min else None
                        salary_max = float(salary_max) if salary_max else None
                    except (ValueError, TypeError):
                        salary_min = salary_max = None

                date_posted = pos.get("PublicationStartDate", "")
                try:
                    date_posted = datetime.strptime(date_posted, "%Y-%m-%dT%H:%M:%S.%f").date()
                except (ValueError, TypeError):
                    try:
                        date_posted = datetime.strptime(date_posted, "%Y-%m-%d").date()
                    except (ValueError, TypeError):
                        date_posted = None

                all_jobs.append({
                    "title": pos.get("PositionTitle", ""),
                    "company": pos.get("OrganizationName", ""),
                    "location": location_str,
                    "state": state,
                    "date_posted": date_posted,
                    "source": "usajobs",
                    "job_url": pos.get("PositionURI", ""),
                    "salary_min": salary_min,
                    "salary_max": salary_max,
                    "is_remote": False,
                    "job_type": pos.get("PositionSchedule", [{}])[0].get("Name", "")
                    if pos.get("PositionSchedule") else "",
                    "description": pos.get("QualificationSummary", ""),
                    "search_term": keyword,
                })

            logger.info(f"  USAJobs: found {len(results)} results for '{keyword}'")

        except Exception as e:
            logger.warning(f"  USAJobs error for '{keyword}': {e}")
            continue

    if not all_jobs:
        return pd.DataFrame()

    return pd.DataFrame(all_jobs)
