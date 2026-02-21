"""Fetch job descriptions from URLs for jobs missing descriptions.

Used by the evaluation pipeline to enrich title-only or boilerplate-only
job listings before sending to Claude API for scoring.
"""

import logging
import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

TIMEOUT = 15  # seconds per request

# CSS selectors to try, in priority order (most specific first)
_JOB_DESC_SELECTORS = [
    # Common ATS platforms
    '[class*="job-description"]',
    '[class*="job_description"]',
    '[class*="jobDescription"]',
    '[id*="job-description"]',
    '[id*="job_description"]',
    '[id*="jobDescription"]',
    '[class*="job-details"]',
    '[class*="job_details"]',
    '[class*="jobDetails"]',
    # Greenhouse
    "#content .body",
    "#content",
    # Lever
    ".posting-page .content",
    ".posting-page",
    # Workday
    '[class*="job-posting"]',
    '[data-automation-id="jobPostingDescription"]',
    # iCIMS
    ".iCIMS_JobContent",
    # LinkedIn public job page
    ".description__text",
    ".show-more-less-html__markup",
    # Generic patterns
    '[class*="description"]',
    '[role="main"]',
    "main",
    "article",
]


def _extract_from_html(html: str) -> str:
    """Extract job description text from HTML page."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise elements
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript", "iframe"]):
        tag.decompose()

    # Try selectors in priority order
    for selector in _JOB_DESC_SELECTORS:
        try:
            elements = soup.select(selector)
        except Exception:
            continue
        for elem in elements:
            text = elem.get_text(separator="\n", strip=True)
            if len(text) > 200:
                return text[:10000]

    # Fallback: body text
    body = soup.find("body")
    if body:
        text = body.get_text(separator="\n", strip=True)
        if len(text) > 200:
            return text[:10000]

    return ""


def _is_valid_url(url: str) -> bool:
    """Check if a URL string is non-empty and not NaN."""
    if not url:
        return False
    url = str(url).strip()
    return bool(url) and url.lower() != "nan" and url.startswith("http")


def fetch_description(
    job_url: str,
    job_url_direct: str = "",
) -> Optional[str]:
    """Try to fetch job description from URL(s).

    Tries direct URL first (company career page), then main job URL.
    Returns extracted description text, or None if fetch failed.
    """
    urls_to_try = []

    # Direct URL first (company ATS page — more likely to be scrapable)
    if _is_valid_url(job_url_direct):
        urls_to_try.append(str(job_url_direct).strip())

    # Main job URL second
    if _is_valid_url(job_url):
        urls_to_try.append(str(job_url).strip())

    for url in urls_to_try:
        try:
            resp = requests.get(
                url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True,
            )
            if resp.status_code == 200:
                text = _extract_from_html(resp.text)
                if text and len(text) > 200:
                    logger.debug(f"Fetched description ({len(text)} chars) from {url}")
                    return text
        except requests.exceptions.Timeout:
            logger.debug(f"Timeout fetching {url}")
        except requests.exceptions.ConnectionError:
            logger.debug(f"Connection error fetching {url}")
        except Exception as e:
            logger.debug(f"Error fetching {url}: {e}")

    return None


def fetch_missing_descriptions(
    jobs_df,
    needs_fetch_mask,
    delay: float = 1.5,
    progress_callback=None,
) -> dict:
    """Fetch descriptions for jobs that need them.

    Args:
        jobs_df: DataFrame with job_url, job_url_direct, description columns
        needs_fetch_mask: Boolean mask of rows that need fetching
        delay: Seconds between requests
        progress_callback: optional callable(fetched, total, succeeded)

    Returns:
        Dict mapping DataFrame index -> fetched description text
    """
    indices = jobs_df.index[needs_fetch_mask].tolist()
    total = len(indices)
    fetched = {}
    succeeded = 0

    logger.info(f"Fetching descriptions for {total} jobs with missing/thin descriptions...")

    for i, idx in enumerate(indices):
        row = jobs_df.loc[idx]
        job_url = str(row.get("job_url", ""))
        job_url_direct = str(row.get("job_url_direct", ""))

        desc = fetch_description(job_url, job_url_direct)
        if desc:
            fetched[idx] = desc
            succeeded += 1

        if progress_callback:
            progress_callback(i + 1, total, succeeded)

        # Rate limit
        if i < total - 1:
            time.sleep(delay)

    logger.info(f"Description fetch complete: {succeeded}/{total} succeeded")
    return fetched
