"""JobSpy wrapper: iterates search terms, collects results into a DataFrame."""

import logging
import time

import pandas as pd

from .config import SearchConfig

logger = logging.getLogger(__name__)


def scrape_jobs(config: SearchConfig, site: str = None) -> pd.DataFrame:
    """Run JobSpy scrape_jobs for each search term, concatenate results.

    Args:
        config: Search configuration.
        site: Optional single site to search (e.g. "indeed" or "linkedin").
              If None, searches all sites in config.sites.
    """
    try:
        from jobspy import scrape_jobs as jobspy_scrape
    except ImportError:
        logger.error("python-jobspy is not installed. Run: pip install python-jobspy")
        return pd.DataFrame()

    sites = [site] if site else config.sites
    all_results = []
    hours_old = config.days * 24

    for i, term in enumerate(config.terms):
        if i > 0:
            logger.info(f"Rate limiting: waiting {config.delay_between_searches}s...")
            time.sleep(config.delay_between_searches)

        logger.info(f"Searching for '{term}' on {sites}...")

        try:
            df = jobspy_scrape(
                site_name=sites,
                search_term=term,
                location=config.location,
                country_indeed=config.country_indeed,
                hours_old=hours_old,
                results_wanted=config.results_per_site,
                description_format="markdown",
                verbose=0,
            )

            if df is not None and not df.empty:
                df["search_term"] = term
                all_results.append(df)
                logger.info(f"  Found {len(df)} results for '{term}'")
            else:
                logger.info(f"  No results for '{term}'")

        except Exception as e:
            logger.warning(f"  Error searching for '{term}': {e}")
            continue

    if not all_results:
        logger.warning("No results found across all search terms.")
        return pd.DataFrame()

    combined = pd.concat([r.dropna(axis=1, how="all") for r in all_results], ignore_index=True)
    logger.info(f"Total raw results: {len(combined)}")
    return combined
