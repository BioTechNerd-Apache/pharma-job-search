"""Streamlit web dashboard for browsing job search results and evaluation results."""

import json
import os
import re
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# When Streamlit runs this file directly, the parent package isn't on sys.path
_src_dir = Path(__file__).resolve().parent
_project_root = _src_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

from src.config import build_config, PROJECT_ROOT, DEFAULT_CONFIG_PATH
from src.exporter import get_master_path

st.set_page_config(
    page_title="Pharma/Biotech Job Search",
    page_icon="\U0001f52c",
    layout="wide",
)

# -- Custom CSS for clean light theme polish --
st.markdown("""
<style>
/* ---- Top-level navigation tabs: large pill-style buttons ---- */
.stMainBlockContainer > div > .stTabs > [data-baseweb="tab-list"] {
    gap: 0.25rem;
    border-bottom: 2px solid #E2E8F0;
    padding-bottom: 0;
}
.stMainBlockContainer > div > .stTabs > [data-baseweb="tab-list"] button[role="tab"] {
    font-size: 1.6rem;
    font-weight: 600;
    padding: 0.85rem 2rem;
    border-radius: 8px 8px 0 0;
    border: 1px solid transparent;
    border-bottom: none;
    color: #475569;
    background: transparent;
    transition: all 0.15s ease;
}
.stMainBlockContainer > div > .stTabs > [data-baseweb="tab-list"] button[role="tab"]:hover {
    background: #F1F5F9;
    color: #1E293B;
}
.stMainBlockContainer > div > .stTabs > [data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
    color: #2563EB;
    background: #EFF6FF;
    border-color: #E2E8F0;
    border-bottom: 2px solid #FFFFFF;
    margin-bottom: -2px;
}
/* Active tab underline indicator */
.stMainBlockContainer > div > .stTabs > [data-baseweb="tab-highlight"] {
    background-color: #2563EB;
    height: 3px;
    border-radius: 3px 3px 0 0;
}

/* ---- Section group headers — left border accent ---- */
.stMainBlockContainer h2 {
    border-left: 4px solid #2563EB;
    padding-left: 0.6rem;
    margin-top: 2.2rem;
    margin-bottom: 0.5rem;
    font-size: 1.5rem;
}
/* Subheaders */
.stMainBlockContainer h3 {
    font-size: 1.15rem;
    margin-top: 1.2rem;
}
/* Subtle card-like containers */
.stMainBlockContainer .stExpander {
    border: 1px solid #E2E8F0;
    border-radius: 8px;
}
/* Bump base font size for labels and captions */
.stMainBlockContainer .stMarkdown p,
.stMainBlockContainer label,
.stMainBlockContainer .stCaption p {
    font-size: 0.95rem;
}
/* Nested tabs (e.g. evaluator patterns) keep default sizing */
.stMainBlockContainer .stTabs .stTabs [data-baseweb="tab-list"] button[role="tab"] {
    font-size: 0.9rem;
    font-weight: 500;
    padding: 0.4rem 1rem;
}
/* ---- Selectbox / dropdown: darker background for contrast ---- */
.stSelectbox [data-baseweb="select"] > div {
    background-color: #F1F5F9;
    border: 1px solid #CBD5E1;
}
/* ---- Text inputs: match dropdown styling ---- */
.stTextInput input {
    background-color: #F1F5F9;
    border: 1px solid #CBD5E1;
}
</style>
""", unsafe_allow_html=True)

REVIEWED_PATH = PROJECT_ROOT / "data" / "reviewed.json"


def load_reviewed() -> dict:
    """Load reviewed timestamps from JSON file. Returns {job_url: timestamp_str}."""
    if REVIEWED_PATH.exists():
        try:
            with open(REVIEWED_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_reviewed(reviewed: dict):
    """Save reviewed timestamps to JSON file."""
    REVIEWED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REVIEWED_PATH, "w") as f:
        json.dump(reviewed, f, indent=2)


def mark_reviewed(job_url: str):
    """Mark a job as reviewed with current timestamp."""
    reviewed = load_reviewed()
    reviewed[job_url] = datetime.now().strftime("%Y-%m-%d %H:%M")
    save_reviewed(reviewed)


def mark_unreviewed(job_url: str):
    """Remove reviewed status from a job."""
    reviewed = load_reviewed()
    reviewed.pop(job_url, None)
    save_reviewed(reviewed)


def extract_job_code(url: str, source: str) -> str:
    """Extract a job code/ID from the job URL based on the source board."""
    if not url or not isinstance(url, str):
        return ""
    try:
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")

        if "linkedin.com" in (parsed.netloc or "") or source == "linkedin":
            match = re.search(r"/jobs/view/(\d+)", path)
            if match:
                return f"LI-{match.group(1)}"

        if "indeed.com" in (parsed.netloc or "") or source == "indeed":
            params = parse_qs(parsed.query)
            if "jk" in params:
                return f"IN-{params['jk'][0]}"
            if "rx_jobId" in params:
                return f"IN-{params['rx_jobId'][0]}"
            segments = path.split("/")
            if segments:
                last = segments[-1]
                match = re.search(r"([a-f0-9]{16,})$", last)
                if match:
                    return f"IN-{match.group(1)[:12]}"

        if "ziprecruiter.com" in (parsed.netloc or "") or source == "zip_recruiter":
            match = re.search(r"/job/([^/?]+)", path) or re.search(r"/jobs/([^/?]+)", path)
            if match:
                return f"ZR-{match.group(1)[:20]}"

        if "glassdoor.com" in (parsed.netloc or "") or source == "glassdoor":
            params = parse_qs(parsed.query)
            if "jobListingId" in params:
                return f"GD-{params['jobListingId'][0]}"

        if source == "google":
            params = parse_qs(parsed.query)
            if "htidocid" in params:
                return f"GJ-{params['htidocid'][0][:12]}"

        if source == "adzuna":
            match = re.search(r"/(\d+)\b", path)
            if match:
                return f"AZ-{match.group(1)}"

        if source == "jooble":
            match = re.search(r"/(\d+)", path)
            if match:
                return f"JB-{match.group(1)}"

        if path:
            last_seg = path.split("/")[-1]
            if last_seg:
                return last_seg[:20]

    except Exception:
        pass
    return ""


def load_data() -> pd.DataFrame:
    """Load data from the master CSV file."""
    config = build_config()
    master_path = get_master_path(config.output, "csv")

    if not master_path.exists():
        # Fallback: try old timestamped files
        data_dir = PROJECT_ROOT / config.output.directory
        if data_dir.exists():
            csv_files = sorted(data_dir.glob(f"{config.output.filename_prefix}_*.csv"), reverse=True)
            if csv_files:
                df = pd.read_csv(csv_files[0], parse_dates=["date_posted"])
                mtime = datetime.fromtimestamp(os.path.getmtime(csv_files[0]))
                st.sidebar.info(f"Last search: {mtime.strftime('%b %d, %Y at %I:%M %p')}")
                return df
        return pd.DataFrame()

    df = pd.read_csv(master_path, parse_dates=["date_posted"])
    mtime = datetime.fromtimestamp(os.path.getmtime(master_path))
    st.sidebar.success(f"Last search: {mtime.strftime('%b %d, %Y at %I:%M %p')} ({len(df)} jobs)")
    return df


def run_search_from_dashboard(search_days=None):
    """Run a new search from the dashboard with progress tracking."""
    from src.aggregator import run_search

    cli_args = {}
    if search_days is not None:
        cli_args["days"] = search_days
    config = build_config(cli_args)

    status = st.status("Searching all job boards in parallel...", expanded=True)
    progress_bar = status.progress(0.0, text="Starting 5 scrapers in parallel...")
    status.write("Scrapers launched: Indeed, LinkedIn, USAJobs, Adzuna, Jooble")
    status.write("Each scraper saves results to CSV as soon as it finishes.")

    completed_count = 0
    total_scrapers = 5
    total_results = 0

    def progress_callback(scraper_name, scraper_status, count=0):
        nonlocal completed_count, total_results
        if scraper_status == "done":
            completed_count += 1
            total_results += count
            pct = completed_count / total_scrapers
            remaining = total_scrapers - completed_count
            status.write(f"  {scraper_name} done — {count} results (saved to CSV)")
            progress_bar.progress(pct, text=f"{completed_count}/{total_scrapers} scrapers done ({remaining} remaining)...")

    # Run the search — each scraper saves to CSV progressively
    df = run_search(config, progress_callback=progress_callback)

    if df.empty:
        status.update(label="Search complete — no results found.", state="error")
        return

    status.update(
        label=f"Search complete! {len(df)} total jobs in master CSV.",
        state="complete",
    )


def run_evaluation_from_dashboard(eval_days=1):
    """Run the evaluation pipeline from the dashboard with progress tracking."""
    from src.evaluator import (
        load_jobs_csv, filter_jobs_by_time,
        run_evaluation_pipeline, estimate_cost,
    )
    from src.eval_persistence import EvaluationStore

    config = build_config()
    store = EvaluationStore(config.evaluation)

    status = st.status("Running job evaluation...", expanded=True)

    # Load master CSV
    df = load_jobs_csv(config)
    if df.empty:
        status.update(label="No master CSV found. Run a search first.", state="error")
        return

    # Filter by time
    filtered = filter_jobs_by_time(df, eval_days=eval_days,
                                   default_days=config.evaluation.default_days)
    if filtered.empty:
        status.update(label="No jobs match the time filter.", state="error")
        return

    # Check how many actually need evaluation
    est = estimate_cost(filtered, config.evaluation, store)
    if est["to_evaluate"] == 0:
        status.update(
            label=f"All {est['already_evaluated']} jobs in this window are already evaluated "
                  f"({est['prefilter_skip']} pre-filter skipped).",
            state="complete",
        )
        return

    cost_str = (f"${est['estimated_cost_usd']:.4f}"
                if isinstance(est["estimated_cost_usd"], (int, float))
                else est["estimated_cost_usd"])
    status.write(f"**{est['total_jobs']}** jobs in window | "
                 f"**{est['prefilter_skip']}** pre-filter skip | "
                 f"**{est['already_evaluated']}** already evaluated | "
                 f"**{est['to_evaluate']}** to evaluate (est. {cost_str})")

    # Progress bars for description fetching stage
    desc_bar = status.progress(0.0, text="Fetching missing job descriptions...")
    sidebar_progress = st.sidebar.progress(0.0, text="Starting evaluation pipeline...")

    def desc_progress_callback(fetched, total_fetch, succeeded):
        pct = fetched / total_fetch if total_fetch else 1.0
        desc_bar.progress(pct, text=f"Fetching descriptions: {fetched}/{total_fetch} ({succeeded} succeeded)")
        sidebar_progress.progress(pct, text=f"Fetching descriptions: {fetched}/{total_fetch}")

    # Progress bars for AI evaluation stage
    eval_bar = status.progress(0.0, text="Waiting to start AI evaluation...")

    def progress_callback(completed, total):
        pct = completed / total if total else 1.0
        eval_bar.progress(pct, text=f"AI evaluation: {completed}/{total} jobs...")
        sidebar_progress.progress(pct, text=f"Evaluating {completed}/{total} jobs...")

    summary = run_evaluation_pipeline(
        jobs_df=filtered,
        config=config.evaluation,
        store=store,
        progress_callback=progress_callback,
        desc_progress_callback=desc_progress_callback,
    )

    status.update(
        label=(f"Evaluation complete! {summary['evaluated']} evaluated, "
               f"{summary['prefilter_skipped']} pre-filter skipped, "
               f"{summary['already_evaluated']} already done."),
        state="complete",
    )


# ---------------------------------------------------------------------------
# Shared AG Grid helpers
# ---------------------------------------------------------------------------

DATE_COMPARATOR = JsCode("""
    function(filterLocalDateAtMidnight, cellValue) {
        if (!cellValue) return -1;
        var parts = cellValue.split('-');
        var cellDate = new Date(Number(parts[0]), Number(parts[1]) - 1, Number(parts[2]));
        if (filterLocalDateAtMidnight.getTime() === cellDate.getTime()) return 0;
        return cellDate < filterLocalDateAtMidnight ? -1 : 1;
    }
""")

LINK_RENDERER = JsCode("""
    class LinkRenderer {
        init(params) {
            this.eGui = document.createElement('a');
            this.eGui.innerText = 'View Job';
            this.eGui.setAttribute('href', params.value);
            this.eGui.setAttribute('target', '_blank');
            this.eGui.style.color = '#1a73e8';
            this.eGui.style.textDecoration = 'underline';
        }
        getGui() { return this.eGui; }
    }
""")

DIRECT_LINK_RENDERER = JsCode("""
    class DirectLinkRenderer {
        init(params) {
            this.eGui = document.createElement('span');
            if (params.value && params.value !== 'nan' && params.value !== '') {
                var a = document.createElement('a');
                a.innerText = 'Apply Direct';
                a.setAttribute('href', params.value);
                a.setAttribute('target', '_blank');
                a.style.color = '#e67c00';
                a.style.textDecoration = 'underline';
                this.eGui.appendChild(a);
            }
        }
        getGui() { return this.eGui; }
    }
""")

TITLE_ONLY_RENDERER = JsCode("""
    class TitleOnlyRenderer {
        init(params) {
            this.eGui = document.createElement('span');
            if (params.value === false || params.value === 'False') {
                this.eGui.innerHTML = '\u26a0\ufe0f Title Only';
                this.eGui.style.color = '#e67c00';
                this.eGui.style.fontWeight = 'bold';
            } else {
                this.eGui.innerHTML = '\u2713 Full';
                this.eGui.style.color = '#28a745';
            }
        }
        getGui() { return this.eGui; }
    }
""")

FIT_SCORE_STYLE = JsCode("""
    function(params) {
        if (params.value >= 70) return {backgroundColor: '#d4edda', fontWeight: 'bold'};
        if (params.value >= 55) return {backgroundColor: '#fff3cd'};
        if (params.value >= 40) return {backgroundColor: '#ffeaa7'};
        return {backgroundColor: '#f8d7da'};
    }
""")


# ---------------------------------------------------------------------------
# Tab 1: Job Listings
# ---------------------------------------------------------------------------

def render_job_listings_tab(df: pd.DataFrame, reviewed_data: dict):
    """Render the Job Listings tab (original dashboard)."""

    # Track grid version to force fresh render after review actions
    if "grid_version" not in st.session_state:
        st.session_state.grid_version = 0

    # --- Sidebar Filters ---
    st.sidebar.header("Job Listings Filters")

    # Unevaluated filter: hide jobs already evaluated or skipped
    if "eval_status" in df.columns:
        unevaluated_count = (df["eval_status"] == "").sum()
        show_unevaluated_only = st.sidebar.checkbox(
            f"Unevaluated only ({unevaluated_count} remaining)", value=True, key="jl_unevaluated"
        )
        if show_unevaluated_only:
            df = df[df["eval_status"] == ""]

    reviewed_count = (df["reviewed_at"] != "").sum()
    unreviewed_count = len(df) - reviewed_count
    show_unreviewed_only = st.sidebar.checkbox(
        f"Unreviewed only ({unreviewed_count} remaining)", value=True, key="jl_unreviewed"
    )
    if show_unreviewed_only:
        df = df[df["reviewed_at"] == ""]

    if "source" in df.columns:
        sources = sorted(df["source"].dropna().unique())
        selected_sources = st.sidebar.multiselect("Job Board", sources, default=sources, key="jl_sources")
        if selected_sources:
            df = df[df["source"].isin(selected_sources)]

    if "state" in df.columns:
        states = sorted(df["state"].dropna().unique())
        states = [s for s in states if s]
        if states:
            selected_states = st.sidebar.multiselect("State", states, key="jl_states")
            if selected_states:
                df = df[df["state"].isin(selected_states)]

    if "is_remote" in df.columns:
        remote_option = st.sidebar.radio("Remote", ["All", "Remote Only", "On-site Only"], key="jl_remote")
        if remote_option == "Remote Only":
            df = df[df["is_remote"] == True]
        elif remote_option == "On-site Only":
            df = df[df["is_remote"] != True]

    if "salary_min" in df.columns:
        has_salary = df["salary_min"].notna().any()
        if has_salary:
            show_salary_only = st.sidebar.checkbox("Only show jobs with salary info", key="jl_salary")
            if show_salary_only:
                df = df[df["salary_min"].notna() | df["salary_max"].notna()]

    st.sidebar.markdown("---")
    show_reposts_only = st.sidebar.checkbox("Only show reposted jobs", key="jl_reposts")
    if show_reposts_only:
        df = df[df["reposted_date"] != ""]

    # --- Main Content ---
    st.markdown(f"**{len(df)} jobs found**")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Jobs", len(df))
    with col2:
        if "source" in df.columns:
            st.metric("Sources", df["source"].nunique())
    with col3:
        if "state" in df.columns:
            st.metric("States", df["state"].replace("", pd.NA).dropna().nunique())
    with col4:
        repost_count = (df["reposted_date"] != "").sum()
        st.metric("Reposted", repost_count)
    with col5:
        st.metric("Reviewed", reviewed_count)

    if "source" in df.columns:
        with st.expander("Source Breakdown"):
            source_counts = df["source"].value_counts()
            st.bar_chart(source_counts)

    if "state" in df.columns:
        with st.expander("Top States"):
            state_counts = df["state"].replace("", pd.NA).dropna().value_counts().head(15)
            st.bar_chart(state_counts)

    # --- AG Grid Table ---
    st.subheader("Job Listings")
    st.caption("Select rows with checkboxes, then click a review button.")

    display_cols = ["job_code", "title", "company", "location", "state",
                    "date_posted", "reposted_date", "days_since_posted",
                    "source", "job_url", "job_url_direct", "salary_min", "salary_max", "is_remote", "job_type",
                    "reviewed_at"]
    display_cols = [c for c in display_cols if c in df.columns]
    grid_df = df[display_cols].copy()
    # Deterministic sort prevents visual rearrangement on selectionChanged reruns
    sort_cols = [c for c in ["date_posted", "job_url"] if c in grid_df.columns]
    if sort_cols:
        grid_df = grid_df.sort_values(sort_cols, ascending=[False, True], na_position="last").reset_index(drop=True)

    gb = GridOptionsBuilder.from_dataframe(grid_df)
    gb.configure_default_column(filterable=True, sortable=True, resizable=True, filter=True)

    for col in ["job_code", "title", "company", "location", "state", "source", "job_type",
                "reposted_date", "reviewed_at"]:
        if col in display_cols:
            gb.configure_column(col, filter="agTextColumnFilter")

    if "date_posted" in display_cols:
        gb.configure_column("date_posted", filter="agDateColumnFilter",
                            filterParams={"comparator": DATE_COMPARATOR})

    for col in ["salary_min", "salary_max", "days_since_posted"]:
        if col in display_cols:
            gb.configure_column(col, filter="agNumberColumnFilter")

    if "job_url" in display_cols:
        gb.configure_column("job_url", headerName="Job Link", cellRenderer=LINK_RENDERER,
                            filter="agTextColumnFilter")

    if "job_url_direct" in display_cols:
        gb.configure_column("job_url_direct", headerName="Direct Link",
                            cellRenderer=DIRECT_LINK_RENDERER, filter="agTextColumnFilter")

    if "is_remote" in display_cols:
        gb.configure_column("is_remote", filter="agSetColumnFilter")

    header_map = {
        "job_code": "Job Code", "title": "Title", "company": "Company",
        "location": "Location", "state": "State", "date_posted": "Posted",
        "reposted_date": "Reposted Date(s)", "days_since_posted": "Days Old",
        "source": "Source", "salary_min": "Salary Min", "salary_max": "Salary Max",
        "is_remote": "Remote", "job_type": "Job Type", "reviewed_at": "Reviewed At",
    }
    for col, name in header_map.items():
        if col in display_cols:
            gb.configure_column(col, headerName=name)

    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=50)
    gb.configure_selection("multiple", use_checkbox=True)
    grid_options = gb.build()
    grid_options["defaultColDef"]["floatingFilter"] = True
    grid_options["defaultColDef"]["suppressSizeToFit"] = True
    grid_options["suppressColumnVirtualisation"] = True
    grid_options["getRowId"] = JsCode("function(params) { return params.data.job_url; }")

    grid_response = AgGrid(
        grid_df, gridOptions=grid_options, update_on=["selectionChanged"],
        allow_unsafe_jscode=True, theme="streamlit", height=600,
        key=f"job_grid_{st.session_state.grid_version}",
    )

    # Capture selection into session_state immediately (before any rerun shifts rows)
    selected = grid_response.get("selected_rows", None)
    selected_rows = []
    if selected is not None:
        if hasattr(selected, "iterrows"):
            selected_rows = [row.to_dict() for _, row in selected.iterrows()]
        elif isinstance(selected, list):
            selected_rows = selected

    if selected_rows:
        st.session_state.jl_selected = selected_rows
    else:
        st.session_state.jl_selected = []

    # Use session_state selections for all actions (immune to grid rearrangement)
    stored = st.session_state.get("jl_selected", [])
    stored_urls = [r.get("job_url", "") for r in stored if r.get("job_url", "")]

    # Review buttons
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 6])
    with btn_col1:
        do_review = st.button(
            f"Mark as Reviewed ({len(stored_urls)})" if stored_urls else "Mark as Reviewed",
            key="jl_review_selected", type="primary",
        )
    with btn_col2:
        do_unreview = st.button(
            f"Undo Review ({len(stored_urls)})" if stored_urls else "Undo Review",
            key="jl_unreview_selected",
        )

    if do_review or do_unreview:
        if not stored_urls:
            st.warning("No rows selected. Check the boxes next to jobs first.")
        else:
            for u in stored_urls:
                if do_review:
                    mark_reviewed(u)
                else:
                    mark_unreviewed(u)
            st.session_state.jl_selected = []
            st.session_state.grid_version += 1
            st.rerun()

    if len(stored) == 1:
        row = stored[0]
        st.subheader("Selected Job Details")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Title:** {row.get('title', 'N/A')}")
            st.markdown(f"**Company:** {row.get('company', 'N/A')}")
            st.markdown(f"**Location:** {row.get('location', 'N/A')}")
            st.markdown(f"**Source:** {row.get('source', 'N/A')}")
        with col_b:
            st.markdown(f"**Job Code:** {row.get('job_code', 'N/A')}")
            st.markdown(f"**Posted:** {row.get('date_posted', 'N/A')}")
            days = row.get("days_since_posted")
            if pd.notna(days):
                st.markdown(f"**Days Old:** {int(days)}")
            repost = row.get("reposted_date", "")
            if repost:
                st.markdown(f"**Reposted on:** {repost}")
            reviewed_ts = row.get("reviewed_at", "")
            if reviewed_ts:
                st.markdown(f"**Reviewed:** {reviewed_ts}")
        url = row.get("job_url", "")
        direct_url = row.get("job_url_direct", "")
        link_parts = []
        if url:
            link_parts.append(f"[View on Job Board]({url})")
        if direct_url and str(direct_url) not in ("", "nan"):
            link_parts.append(f"[Apply Direct (Employer Site)]({direct_url})")
        if link_parts:
            st.markdown(" | ".join(link_parts))
    elif len(stored) > 1:
        st.info(f"{len(stored)} jobs selected — click a review button above.")

    # Expandable job descriptions
    st.subheader("Job Descriptions")
    if "description" in df.columns:
        for idx, row in df.head(50).iterrows():
            title = row.get("title", "Untitled")
            company = row.get("company", "Unknown")
            job_code = row.get("job_code", "")
            label = f"[{job_code}] {title} - {company}" if job_code else f"{title} - {company}"
            desc = row.get("description", "")
            if pd.notna(desc) and desc:
                with st.expander(label):
                    st.markdown(f"**Location:** {row.get('location', 'N/A')}")
                    st.markdown(f"**Source:** {row.get('source', 'N/A')}")
                    days = row.get("days_since_posted")
                    if pd.notna(days):
                        st.markdown(f"**Days old:** {int(days)}")
                    repost = row.get("reposted_date", "")
                    if repost:
                        st.markdown(f"**Reposted on:** {repost}")
                    link_parts = []
                    if pd.notna(row.get("job_url")):
                        link_parts.append(f"[View on Job Board]({row['job_url']})")
                    direct = row.get("job_url_direct", "")
                    if pd.notna(direct) and str(direct) not in ("", "nan"):
                        link_parts.append(f"[Apply Direct]({direct})")
                    if link_parts:
                        st.markdown(" | ".join(link_parts))
                    st.markdown("---")
                    st.markdown(str(desc)[:3000])


# ---------------------------------------------------------------------------
# Tab 2: Evaluation Results
# ---------------------------------------------------------------------------

def load_evaluation_data(df: pd.DataFrame) -> pd.DataFrame:
    """Load evaluation results and merge with job data."""
    config = build_config()
    eval_path = PROJECT_ROOT / config.evaluation.evaluations_store

    if not eval_path.exists():
        return pd.DataFrame()

    try:
        with open(eval_path, "r") as f:
            eval_data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return pd.DataFrame()

    evals = eval_data.get("evaluations", {})
    if not evals:
        return pd.DataFrame()

    eval_records = []
    for url, ev in evals.items():
        record = {"job_url": url}
        record.update(ev)
        eval_records.append(record)

    eval_df = pd.DataFrame(eval_records)

    if df.empty or eval_df.empty:
        return eval_df

    merged = df.merge(eval_df, on="job_url", how="inner")
    return merged


def render_evaluation_tab(df: pd.DataFrame, reviewed_data: dict):
    """Render the Evaluation Results tab."""

    if "eval_grid_version" not in st.session_state:
        st.session_state.eval_grid_version = 0

    eval_df = load_evaluation_data(df)

    if eval_df.empty:
        st.warning("No evaluation results found. Run evaluation first:\n\n"
                    "`python job_search.py --evaluate-only --eval-days 1`")
        return

    # Merge reviewed timestamps
    eval_df["reviewed_at"] = eval_df["job_url"].map(reviewed_data).fillna("")

    # --- Sidebar Filters ---
    st.sidebar.header("Evaluation Filters")

    # Precompute stable option lists from UNFILTERED data so Streamlit multiselect
    # widgets always see the same options between reruns (prevents ghost rows).
    all_recs = sorted(eval_df["recommendation"].dropna().unique()) if "recommendation" in eval_df.columns else []
    all_buckets = sorted(eval_df["fit_bucket"].dropna().unique()) if "fit_bucket" in eval_df.columns else []
    all_sources = sorted(eval_df["source"].dropna().unique()) if "source" in eval_df.columns else []

    min_score = st.sidebar.slider("Min Fit Score", 0, 100, 0, key="eval_min_score")
    if min_score > 0:
        eval_df = eval_df[eval_df["fit_score"] >= min_score]

    if all_recs:
        selected_recs = st.sidebar.multiselect("Recommendation", all_recs, default=all_recs, key="eval_recs")
        if selected_recs:
            eval_df = eval_df[eval_df["recommendation"].isin(selected_recs)]

    if all_buckets:
        selected_buckets = st.sidebar.multiselect("Fit Bucket", all_buckets, default=all_buckets, key="eval_buckets")
        if selected_buckets:
            eval_df = eval_df[eval_df["fit_bucket"].isin(selected_buckets)]

    reviewed_count = (eval_df["reviewed_at"] != "").sum()
    unreviewed_count = len(eval_df) - reviewed_count
    show_unreviewed = st.sidebar.checkbox(
        f"Unreviewed only ({unreviewed_count} remaining)", value=False, key="eval_unreviewed"
    )
    if show_unreviewed:
        eval_df = eval_df[eval_df["reviewed_at"] == ""]

    if "description_available" in eval_df.columns:
        title_only_count = (~eval_df["description_available"].astype(bool)).sum()
        show_title_only = st.sidebar.checkbox(
            f"Title-only jobs only ({title_only_count})", value=False, key="eval_title_only"
        )
        if show_title_only:
            eval_df = eval_df[~eval_df["description_available"].astype(bool)]

    if all_sources:
        selected_sources = st.sidebar.multiselect("Source", all_sources, default=all_sources, key="eval_sources")
        if selected_sources:
            eval_df = eval_df[eval_df["source"].isin(selected_sources)]

    if "state" in eval_df.columns:
        states = sorted(eval_df["state"].dropna().unique())
        states = [s for s in states if s]
        if states:
            selected_states = st.sidebar.multiselect("State", states, key="eval_states")
            if selected_states:
                eval_df = eval_df[eval_df["state"].isin(selected_states)]

    if "is_remote" in eval_df.columns:
        remote_option = st.sidebar.radio("Remote", ["All", "Remote Only", "On-site Only"], key="eval_remote")
        if remote_option == "Remote Only":
            eval_df = eval_df[eval_df["is_remote"] == True]
        elif remote_option == "On-site Only":
            eval_df = eval_df[eval_df["is_remote"] != True]

    # --- Summary Metrics ---
    total_eval = len(eval_df)
    apply_count = len(eval_df[eval_df.get("recommendation", pd.Series()) == "apply"]) if "recommendation" in eval_df.columns else 0
    maybe_count = len(eval_df[eval_df.get("recommendation", pd.Series()) == "maybe"]) if "recommendation" in eval_df.columns else 0
    skip_count = len(eval_df[eval_df.get("recommendation", pd.Series()) == "skip"]) if "recommendation" in eval_df.columns else 0
    avg_score = round(eval_df["fit_score"].mean(), 1) if "fit_score" in eval_df.columns and total_eval > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Evaluated", total_eval)
    with c2:
        st.metric("Apply", apply_count)
    with c3:
        st.metric("Maybe", maybe_count)
    with c4:
        st.metric("Skip", skip_count)
    with c5:
        st.metric("Avg Score", avg_score)

    # --- AG Grid ---
    st.subheader("Evaluation Results")

    # Prepare display columns
    display_cols = [
        "fit_score", "fit_bucket", "recommendation",
        "description_available",
        "title", "company", "domain_match",
        "location", "state", "date_posted", "days_since_posted",
        "source", "job_url", "job_url_direct",
        "evaluated_timestamp", "reviewed_at",
    ]
    display_cols = [c for c in display_cols if c in eval_df.columns]

    # Compute days_since_posted if not present
    if "days_since_posted" not in eval_df.columns and "date_posted" in eval_df.columns:
        eval_df["date_posted"] = pd.to_datetime(eval_df["date_posted"], errors="coerce")
        today = pd.Timestamp.now().normalize()
        eval_df["days_since_posted"] = (today - eval_df["date_posted"]).dt.days
        eval_df["date_posted"] = eval_df["date_posted"].dt.strftime("%Y-%m-%d")
        if "days_since_posted" not in display_cols:
            display_cols.insert(display_cols.index("date_posted") + 1, "days_since_posted")

    grid_df = eval_df[display_cols].copy()

    # Deterministic sort prevents visual rearrangement on selectionChanged reruns
    sort_cols = [c for c in ["fit_score", "job_url"] if c in grid_df.columns]
    if sort_cols:
        grid_df = grid_df.sort_values(sort_cols, ascending=[False, True], na_position="last").reset_index(drop=True)

    gb = GridOptionsBuilder.from_dataframe(grid_df)
    gb.configure_default_column(filterable=True, sortable=True, resizable=True, filter=True)

    # Fit score with color coding
    if "fit_score" in display_cols:
        gb.configure_column("fit_score", headerName="Fit Score",
                            filter="agNumberColumnFilter", cellStyle=FIT_SCORE_STYLE,
                            filterParams={"defaultOption": "greaterThanOrEqual",
                                          "suppressAndOrCondition": True})

    if "fit_bucket" in display_cols:
        gb.configure_column("fit_bucket", headerName="Fit Bucket", filter="agSetColumnFilter")

    if "recommendation" in display_cols:
        gb.configure_column("recommendation", headerName="Recommendation", filter="agSetColumnFilter")

    if "description_available" in display_cols:
        gb.configure_column("description_available", headerName="Info",
                            cellRenderer=TITLE_ONLY_RENDERER, filter="agSetColumnFilter")

    for col in ["title", "company", "domain_match", "location", "state",
                "evaluated_timestamp", "reviewed_at"]:
        if col in display_cols:
            gb.configure_column(col, filter="agTextColumnFilter")

    if "date_posted" in display_cols:
        gb.configure_column("date_posted", headerName="Posted",
                            filter="agDateColumnFilter",
                            filterParams={"comparator": DATE_COMPARATOR})

    for col in ["days_since_posted"]:
        if col in display_cols:
            gb.configure_column(col, headerName="Days Old", filter="agNumberColumnFilter",
                                filterParams={"defaultOption": "greaterThanOrEqual",
                                              "suppressAndOrCondition": True})

    if "source" in display_cols:
        gb.configure_column("source", headerName="Source", filter="agSetColumnFilter")

    if "job_url" in display_cols:
        gb.configure_column("job_url", headerName="Job Link",
                            cellRenderer=LINK_RENDERER, filter="agTextColumnFilter")

    if "job_url_direct" in display_cols:
        gb.configure_column("job_url_direct", headerName="Direct Link",
                            cellRenderer=DIRECT_LINK_RENDERER, filter="agTextColumnFilter")

    # Human-friendly header names
    header_map = {
        "title": "Title", "company": "Company", "domain_match": "Domain Match",
        "location": "Location", "state": "State",
        "evaluated_timestamp": "Evaluated At", "reviewed_at": "Reviewed At",
    }
    for col, name in header_map.items():
        if col in display_cols:
            gb.configure_column(col, headerName=name)

    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=50)
    gb.configure_selection("multiple", use_checkbox=True)
    grid_options = gb.build()
    grid_options["defaultColDef"]["floatingFilter"] = True
    grid_options["defaultColDef"]["unSortIcon"] = True
    grid_options["defaultColDef"]["suppressSizeToFit"] = True
    grid_options["suppressColumnVirtualisation"] = True
    grid_options["getRowId"] = JsCode("function(params) { return params.data.job_url; }")

    grid_response = AgGrid(
        grid_df, gridOptions=grid_options, update_on=["selectionChanged"],
        allow_unsafe_jscode=True, theme="streamlit", height=600,
        key=f"eval_grid_{st.session_state.eval_grid_version}",
    )

    # Capture selection into session_state immediately (before any rerun shifts rows)
    selected = grid_response.get("selected_rows", None)
    selected_rows = []
    if selected is not None:
        if hasattr(selected, "iterrows"):
            selected_rows = [row.to_dict() for _, row in selected.iterrows()]
        elif isinstance(selected, list):
            selected_rows = selected

    if selected_rows:
        st.session_state.eval_selected = selected_rows
    else:
        st.session_state.eval_selected = []

    # Use session_state selections for all actions (immune to grid rearrangement)
    stored = st.session_state.get("eval_selected", [])
    stored_urls = [r.get("job_url", "") for r in stored if r.get("job_url", "")]

    # Review buttons
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 6])
    with btn_col1:
        do_review = st.button(
            f"Mark as Reviewed ({len(stored_urls)})" if stored_urls else "Mark as Reviewed",
            key="eval_review", type="primary",
        )
    with btn_col2:
        do_unreview = st.button(
            f"Undo Review ({len(stored_urls)})" if stored_urls else "Undo Review",
            key="eval_unreview",
        )

    if do_review or do_unreview:
        if not stored_urls:
            st.warning("No rows selected. Check the boxes next to jobs first.")
        else:
            for u in stored_urls:
                if do_review:
                    mark_reviewed(u)
                else:
                    mark_unreviewed(u)
            st.session_state.eval_selected = []
            st.session_state.eval_grid_version += 1
            st.rerun()

    # Detail panel for single selected row
    if len(stored) == 1:
        row = stored[0]
        st.subheader("Evaluation Details")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Title:** {row.get('title', 'N/A')}")
            st.markdown(f"**Company:** {row.get('company', 'N/A')}")
            st.markdown(f"**Location:** {row.get('location', 'N/A')}")
            score = row.get('fit_score', 0)
            bucket = row.get('fit_bucket', 'N/A')
            rec = row.get('recommendation', 'N/A')
            st.markdown(f"**Fit Score:** {score} ({bucket}) — **{rec.upper()}**")
        with col_b:
            st.markdown(f"**Domain Match:** {row.get('domain_match', 'N/A')}")
            st.markdown(f"**Source:** {row.get('source', 'N/A')}")
            st.markdown(f"**Posted:** {row.get('date_posted', 'N/A')}")
            reviewed_ts = row.get("reviewed_at", "")
            if reviewed_ts:
                st.markdown(f"**Reviewed:** {reviewed_ts}")

        # Reasoning
        reasoning = row.get("reasoning", "")
        if reasoning:
            st.markdown(f"**Reasoning:** {reasoning}")

        # Links
        url = row.get("job_url", "")
        direct_url = row.get("job_url_direct", "")
        link_parts = []
        if url:
            link_parts.append(f"[View on Job Board]({url})")
        if direct_url and str(direct_url) not in ("", "nan"):
            link_parts.append(f"[Apply Direct (Employer Site)]({direct_url})")
        if link_parts:
            st.markdown(" | ".join(link_parts))

        # Job description if available
        if "description" in eval_df.columns:
            job_row = eval_df[eval_df["job_url"] == url]
            if not job_row.empty:
                desc = job_row.iloc[0].get("description", "")
                if pd.notna(desc) and str(desc).strip() and str(desc).lower() != "nan":
                    with st.expander("Full Job Description"):
                        st.markdown(str(desc)[:5000])

    elif len(stored) > 1:
        st.info(f"{len(stored)} jobs selected — click a review button above.")


# ---------------------------------------------------------------------------
# Tab 3: Setup / Profile
# ---------------------------------------------------------------------------

def _load_resume_profile() -> dict:
    """Load resume profile JSON."""
    path = PROJECT_ROOT / "data" / "resume_profile.json"
    if path.is_file():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_resume_profile(data: dict):
    """Save resume profile JSON."""
    path = PROJECT_ROOT / "data" / "resume_profile.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_evaluator_patterns() -> dict:
    """Load evaluator patterns YAML."""
    import yaml
    path = PROJECT_ROOT / "data" / "evaluator_patterns.yaml"
    if path.is_file():
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    return {}


def _save_evaluator_patterns(data: dict):
    """Save evaluator patterns YAML."""
    import yaml
    path = PROJECT_ROOT / "data" / "evaluator_patterns.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _load_config_yaml() -> dict:
    """Load config.yaml."""
    import yaml
    if DEFAULT_CONFIG_PATH.is_file():
        with open(DEFAULT_CONFIG_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_config_yaml(data: dict):
    """Save config.yaml."""
    import yaml
    with open(DEFAULT_CONFIG_PATH, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _resolve_wizard_ai(config_yaml: dict) -> tuple[str, str]:
    """Resolve the effective provider/model the wizard will use.

    Returns (provider, model) — checks wizard config first, falls back to evaluation.
    """
    from src.ai_client import DEFAULT_MODELS
    wiz = config_yaml.get("wizard", {})
    if wiz.get("provider"):
        provider = wiz["provider"]
        model = wiz.get("model", "") or DEFAULT_MODELS.get(provider, "")
        return provider, model
    ev = config_yaml.get("evaluation", {})
    provider = ev.get("provider", "anthropic")
    model = ev.get("model", "") or DEFAULT_MODELS.get(provider, "")
    return provider, model


def _render_ai_provider_picker(config_section: str, key_prefix: str) -> None:
    """Render provider/model/api_key/base_url picker for a config section.

    Args:
        config_section: "wizard" or "evaluation" — key in config.yaml
        key_prefix: Streamlit widget key prefix (e.g. "wizard" or "eval")
    """
    from src.ai_client import DEFAULT_MODELS, PROVIDER_MODELS, MODEL_PRICING

    config_yaml = _load_config_yaml()
    section_cfg = config_yaml.get(config_section, {})

    providers = ["anthropic", "openai", "ollama"]
    current_provider = section_cfg.get("provider", "" if config_section == "wizard" else "anthropic")
    if config_section == "wizard":
        provider_options = ["(use evaluation provider)"] + providers
        if current_provider in providers:
            provider_idx = providers.index(current_provider) + 1
        else:
            provider_idx = 0  # "(use evaluation provider)"
    else:
        provider_options = providers
        provider_idx = providers.index(current_provider) if current_provider in providers else 0

    new_provider_display = st.selectbox(
        "Provider", provider_options, index=provider_idx, key=f"{key_prefix}_provider"
    )

    # Resolve actual provider value to store
    if new_provider_display == "(use evaluation provider)":
        new_provider = ""
    else:
        new_provider = new_provider_display

    # Model selection — only show if a provider is selected
    if new_provider:
        default_model = DEFAULT_MODELS.get(new_provider, "")
        current_model = section_cfg.get("model", default_model)

        known_models = PROVIDER_MODELS.get(new_provider, [])
        model_options = known_models + ["Other (custom)"]

        if current_model in known_models:
            model_idx = known_models.index(current_model)
        else:
            model_idx = len(known_models)

        selected_model = st.selectbox("Model", model_options, index=model_idx, key=f"{key_prefix}_model")

        if selected_model == "Other (custom)":
            custom_val = current_model if current_model not in known_models else ""
            new_model = st.text_input("Custom model name", value=custom_val, key=f"{key_prefix}_model_custom")
        else:
            new_model = selected_model

        # Cost estimate
        pricing = MODEL_PRICING.get(new_model)
        if pricing:
            input_rate, output_rate = pricing
            avg_input, avg_output = 2500, 300
            cost_per_call = (avg_input / 1_000_000) * input_rate + (avg_output / 1_000_000) * output_rate
            label = "per wizard call" if config_section == "wizard" else "per job evaluation"
            st.info(
                f"**Estimated cost:** ~${cost_per_call:.4f} {label}  \n"
                f"Input: ${input_rate:.2f} / Output: ${output_rate:.2f} per 1M tokens"
            )
        elif new_provider == "ollama":
            st.info(
                "**Cost:** Free (local model)  \n\n"
                "**Setup required:** Install [Ollama](https://ollama.com/download) "
                "and pull the model:  \n"
                f"`ollama pull {new_model}`  \n\n"
                "Ollama must be running before you can use it here "
                "(start it with `ollama serve` or launch the app)."
            )
        else:
            st.caption("Cost estimate unavailable for custom models.")

        # Base URL
        if new_provider == "ollama":
            default_base = "http://localhost:11434/v1"
            current_base = section_cfg.get("base_url", default_base)
            new_base_url = st.text_input(
                "Base URL", value=current_base, key=f"{key_prefix}_base_url",
                help="Where Ollama's local server listens. "
                     "The default (http://localhost:11434/v1) is correct unless "
                     "you changed Ollama's port.",
                disabled=False,
            )
        elif new_provider == "openai":
            current_base = section_cfg.get("base_url", "")
            if current_base:
                # Show the field if a custom base URL is already saved
                new_base_url = st.text_input(
                    "Base URL (optional)", value=current_base,
                    key=f"{key_prefix}_base_url",
                    help="Leave blank to use the default OpenAI API. "
                         "Only set this for Azure OpenAI, proxies, or OpenAI-compatible APIs.",
                )
            else:
                # Hide behind expander — most OpenAI users don't need this
                with st.expander("Advanced: Custom Base URL"):
                    new_base_url = st.text_input(
                        "Base URL", value="", key=f"{key_prefix}_base_url",
                        help="Leave blank to use the default OpenAI API. "
                             "Only set this for Azure OpenAI, proxies, or OpenAI-compatible APIs.",
                    )
        else:
            new_base_url = ""

        # API Key
        if new_provider != "ollama":
            current_api_key = (
                section_cfg.get("api_key", "")
                or (section_cfg.get("anthropic_api_key", "") if config_section == "evaluation" else "")
                or os.environ.get({
                    "anthropic": "ANTHROPIC_API_KEY",
                    "openai": "OPENAI_API_KEY",
                }.get(new_provider, ""), "")
            )
            new_api_key = st.text_input(
                f"{new_provider.title()} API Key", value=current_api_key,
                type="password", key=f"{key_prefix}_api_key"
            )
        else:
            new_api_key = ""
    else:
        new_model = ""
        new_base_url = ""
        new_api_key = ""

    label = "Save Wizard AI Provider" if config_section == "wizard" else "Save Evaluation AI Provider"
    if st.button(label, key=f"{key_prefix}_save_provider"):
        config_yaml = _load_config_yaml()
        config_yaml.setdefault(config_section, {})
        config_yaml[config_section]["provider"] = new_provider
        config_yaml[config_section]["model"] = new_model
        config_yaml[config_section]["base_url"] = new_base_url
        if new_api_key:
            config_yaml[config_section]["api_key"] = new_api_key
        _save_config_yaml(config_yaml)
        display_provider = new_provider or "(evaluation fallback)"
        display_model = new_model or "(evaluation fallback)"
        st.success(f"Saved: {display_provider} / {display_model}")


def render_setup_tab():
    """Render the Setup / Profile tab."""
    from src.pattern_helpers import regex_to_display, display_to_regex

    # ===================================================================
    # Group 1: AI Providers
    # ===================================================================
    st.header("AI Providers")

    st.subheader("Wizard AI Provider")
    st.caption("Used for resume parsing and generating search config. "
               "If not set, falls back to the Evaluation AI Provider below.")
    _render_ai_provider_picker("wizard", "wizard")

    st.markdown("---")

    st.subheader("Evaluation AI Provider")
    st.caption("Used for scoring jobs against your resume profile.")
    _render_ai_provider_picker("evaluation", "eval")

    # ===================================================================
    # Group 2: Getting Started
    # ===================================================================
    st.header("Getting Started")

    st.subheader("Setup Wizard")
    st.caption("Upload a resume to auto-generate all configuration using AI.")

    # Show which AI provider the wizard will use (reads live dropdown state)
    from src.ai_client import DEFAULT_MODELS
    wiz_provider_display = st.session_state.get("wizard_provider", "")
    if wiz_provider_display and wiz_provider_display != "(use evaluation provider)":
        eff_provider = wiz_provider_display
        eff_model = st.session_state.get("wizard_model", "") or DEFAULT_MODELS.get(eff_provider, "")
    else:
        # Wizard not configured — fall back to evaluation provider dropdown
        eff_provider = st.session_state.get("eval_provider", "")
        eff_model = st.session_state.get("eval_model", "")
        if not eff_provider:
            # First render before widgets exist — read from saved config
            cfg = _load_config_yaml()
            eff_provider, eff_model = _resolve_wizard_ai(cfg)
        elif not eff_model:
            eff_model = DEFAULT_MODELS.get(eff_provider, "")
    if eff_model == "Other (custom)":
        eff_model = st.session_state.get("wizard_model_custom", "") or st.session_state.get("eval_model_custom", "") or "custom"
    st.info(f"Will use: **{eff_provider}** / **{eff_model}**")

    # Warn if setup has already been completed
    has_existing_profile = bool(_load_resume_profile())
    if has_existing_profile:
        st.warning(
            "A resume profile already exists. Running the wizard again will "
            "**overwrite** your current profile, search terms, filters, and "
            "evaluator patterns with new AI-generated configuration.",
            icon="\u26a0\ufe0f",
        )

    uploaded = st.file_uploader(
        "Upload resume (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
        key="setup_wizard_upload"
    )

    # Require explicit confirmation when overwriting existing config
    confirm_overwrite = True
    if has_existing_profile and uploaded:
        confirm_overwrite = st.checkbox(
            "I understand this will replace my existing configuration",
            key="setup_wizard_confirm",
        )

    if uploaded and confirm_overwrite and st.button(
        "Run Setup Wizard", key="setup_run_wizard", type="primary"
    ):
        # Save uploaded file to temp location
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        try:
            from src.resume_parser import extract_text
            with st.status("Running setup wizard...", expanded=True) as status:
                # Step 1: Extract text
                status.write("Extracting resume text...")
                resume_text = extract_text(tmp_path)
                status.write(f"Extracted {len(resume_text):,} characters")

                # Resolve wizard AI: wizard config > evaluation config
                cfg = _load_config_yaml()
                wiz_cfg = cfg.get("wizard", {})
                eval_cfg = cfg.get("evaluation", {})

                if wiz_cfg.get("provider"):
                    provider = wiz_cfg["provider"]
                    model = wiz_cfg.get("model", "")
                    base_url = wiz_cfg.get("base_url", "")
                    api_key = (
                        wiz_cfg.get("api_key", "")
                        or os.environ.get({
                            "anthropic": "ANTHROPIC_API_KEY",
                            "openai": "OPENAI_API_KEY",
                        }.get(provider, ""), "")
                    )
                else:
                    provider = eval_cfg.get("provider", "anthropic")
                    model = eval_cfg.get("model", "")
                    base_url = eval_cfg.get("base_url", "")
                    api_key = (
                        eval_cfg.get("api_key", "")
                        or eval_cfg.get("anthropic_api_key", "")
                        or os.environ.get({
                            "anthropic": "ANTHROPIC_API_KEY",
                            "openai": "OPENAI_API_KEY",
                        }.get(provider, ""), "")
                    )

                if not api_key and provider != "ollama":
                    status.update(label="Setup wizard failed", state="error")
                    st.error(f"API key is required for provider '{provider}'. "
                             "Configure the AI provider in the AI Providers section above.")
                    return

                from src.ai_client import AIClient
                client = AIClient(
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                )

                # Step 2: Generate profile
                status.write(f"Generating resume profile (AI Call 1/3 via {provider})...")
                from src.setup_wizard import (
                    call_generate_profile,
                    call_generate_search_config,
                    call_generate_evaluator_patterns,
                )
                profile_data = call_generate_profile(client, resume_text)
                name = profile_data.get("name", "Unknown")
                status.write(f"Profile generated for: {name}")

                # Step 3: Generate search config
                status.write("Generating search configuration (AI Call 2/3)...")
                search_data = call_generate_search_config(client, profile_data)
                status.write(f"Generated {len(search_data.get('search_terms', []))} search terms")

                # Step 4: Generate patterns
                status.write("Generating evaluator patterns (AI Call 3/3)...")
                pattern_data = call_generate_evaluator_patterns(client, profile_data)
                status.write(f"Generated {len(pattern_data.get('skip_title_patterns', []))} skip patterns")

                # Store results for preview — do NOT save yet
                st.session_state["wizard_preview"] = {
                    "profile": profile_data,
                    "search": search_data,
                    "patterns": pattern_data,
                }
                status.update(label="Generation complete — review below", state="complete")

        except Exception as e:
            st.error(f"Wizard error: {e}")
        finally:
            os.unlink(tmp_path)

    # ===================================================================
    # Wizard Preview / Approval
    # ===================================================================
    if "wizard_preview" in st.session_state:
        preview = st.session_state["wizard_preview"]
        profile_data = preview["profile"]
        search_data = preview["search"]
        pattern_data = preview["patterns"]

        st.subheader("Review AI-Generated Configuration")
        st.info(
            "Review and edit the generated configuration below, then click "
            "**Save Configuration** to apply."
        )

        # Resume profile (read-only)
        with st.expander("Resume Profile (read-only)", expanded=False):
            st.json(profile_data)

        # Search terms + Priority terms
        col_st, col_pt = st.columns(2)
        with col_st:
            search_terms_list = search_data.get("search_terms", [])
            wiz_search_terms = st.text_area(
                f"Search Terms ({len(search_terms_list)} terms, one per line)",
                value="\n".join(search_terms_list),
                height=200,
                key="wiz_search_terms",
            )
        with col_pt:
            priority_list = search_data.get("priority_terms", [])
            wiz_priority_terms = st.text_area(
                f"Priority Terms ({len(priority_list)} terms, one per line)",
                value="\n".join(priority_list),
                height=200,
                key="wiz_priority_terms",
            )

        # Include + Exclude filter keywords
        col_inc, col_exc = st.columns(2)
        with col_inc:
            inc_list = search_data.get("filter_include", [])
            wiz_filter_include = st.text_area(
                f"Include Filter Keywords ({len(inc_list)} keywords, one per line)",
                value="\n".join(inc_list),
                height=200,
                key="wiz_filter_include",
            )
        with col_exc:
            exc_list = search_data.get("filter_exclude", [])
            wiz_filter_exclude = st.text_area(
                f"Exclude Filter Keywords ({len(exc_list)} keywords, one per line)",
                value="\n".join(exc_list),
                height=200,
                key="wiz_filter_exclude",
            )

        # Synonyms
        with st.expander("Synonyms (editable)", expanded=False):
            syn_dict = search_data.get("synonyms", {})
            syn_lines = [f"{k}: {', '.join(v)}" for k, v in syn_dict.items()]
            wiz_synonyms = st.text_area(
                f"Synonyms ({len(syn_dict)} entries — format: term: alias1, alias2)",
                value="\n".join(syn_lines),
                height=200,
                key="wiz_synonyms",
            )

        # Evaluator patterns (read-only)
        with st.expander("Evaluator Patterns (read-only)", expanded=False):
            st.json(pattern_data)

        # Save / Discard buttons
        col_save, col_discard = st.columns(2)
        with col_save:
            if st.button("Save Configuration", type="primary", key="wiz_save"):
                # Parse edited text areas back into lists/dicts
                edited_terms = [
                    t.strip() for t in wiz_search_terms.split("\n") if t.strip()
                ]
                edited_priority = [
                    t.strip() for t in wiz_priority_terms.split("\n") if t.strip()
                ]
                edited_include = [
                    t.strip() for t in wiz_filter_include.split("\n") if t.strip()
                ]
                edited_exclude = [
                    t.strip() for t in wiz_filter_exclude.split("\n") if t.strip()
                ]
                # Parse synonyms: "key: val1, val2" -> {key: [val1, val2]}
                edited_synonyms: dict[str, list[str]] = {}
                for line in wiz_synonyms.split("\n"):
                    line = line.strip()
                    if not line or ":" not in line:
                        continue
                    key, _, vals = line.partition(":")
                    key = key.strip()
                    if key:
                        edited_synonyms[key] = [
                            v.strip() for v in vals.split(",") if v.strip()
                        ]

                from src.setup_wizard import WizardOutput, save_wizard_output

                output = WizardOutput(
                    resume_profile=profile_data,
                    search_terms=edited_terms,
                    priority_terms=edited_priority,
                    synonyms=edited_synonyms,
                    filter_include=edited_include,
                    filter_exclude=edited_exclude,
                    evaluator_patterns=pattern_data,
                )
                cfg = _load_config_yaml()
                files = save_wizard_output(output, cfg)
                del st.session_state["wizard_preview"]
                st.success(
                    f"Configuration saved! {len(files)} files written. "
                    "Reload the page to see updated configuration."
                )
                st.balloons()
                st.rerun()

        with col_discard:
            if st.button("Discard", key="wiz_discard"):
                del st.session_state["wizard_preview"]
                st.rerun()

    # ===================================================================
    # Group 3: Search Configuration
    # ===================================================================
    st.header("Search Configuration")

    # -- Search Terms & Synonyms --
    st.subheader("Search Terms & Synonyms")
    config_yaml = _load_config_yaml()
    search_cfg = config_yaml.get("search", {})

    col_terms, col_syn = st.columns(2)
    with col_terms:
        terms = search_cfg.get("terms", [])
        priority = search_cfg.get("priority_terms", [])
        terms_text = "\n".join(terms) if terms else ""
        priority_text = "\n".join(priority) if priority else ""

        st.caption("Search terms (one per line)")
        edited_terms = st.text_area("Search Terms", value=terms_text, height=200, key="setup_terms")

        st.caption("Priority terms (one per line, subset of search terms)")
        edited_priority = st.text_area("Priority Terms", value=priority_text, height=100, key="setup_priority")

    with col_syn:
        synonyms = search_cfg.get("synonyms", {})
        syn_text = json.dumps(synonyms, indent=2) if synonyms else "{}"
        st.caption("Synonym groups (JSON dict)")
        edited_syn = st.text_area("Synonyms", value=syn_text, height=320, key="setup_synonyms")

    if st.button("Save Search Config", key="setup_save_search"):
        try:
            new_terms = [t.strip() for t in edited_terms.strip().split("\n") if t.strip()]
            new_priority = [t.strip() for t in edited_priority.strip().split("\n") if t.strip()]
            new_syn = json.loads(edited_syn)

            config_yaml.setdefault("search", {})
            config_yaml["search"]["terms"] = new_terms
            config_yaml["search"]["priority_terms"] = new_priority
            config_yaml["search"]["synonyms"] = new_syn
            _save_config_yaml(config_yaml)
            st.success("Search configuration saved to config.yaml.")
        except json.JSONDecodeError as e:
            st.error(f"Invalid synonyms JSON: {e}")

    st.markdown("---")

    # -- Discipline Filters --
    st.subheader("Discipline Filters")
    col_inc, col_exc = st.columns(2)

    with col_inc:
        includes = search_cfg.get("filter_include", [])
        inc_text = "\n".join(includes) if includes else ""
        st.caption("Include keywords (one per line) — title must match at least one")
        edited_inc = st.text_area("Include Filters", value=inc_text, height=300, key="setup_includes")

    with col_exc:
        excludes = search_cfg.get("filter_exclude", [])
        exc_text = "\n".join(excludes) if excludes else ""
        st.caption("Exclude keywords (one per line) — title matching any of these is removed")
        edited_exc = st.text_area("Exclude Filters", value=exc_text, height=300, key="setup_excludes")

    if st.button("Save Filters", key="setup_save_filters"):
        new_inc = [t.strip() for t in edited_inc.strip().split("\n") if t.strip()]
        new_exc = [t.strip() for t in edited_exc.strip().split("\n") if t.strip()]
        config_yaml = _load_config_yaml()
        config_yaml.setdefault("search", {})
        config_yaml["search"]["filter_include"] = new_inc
        config_yaml["search"]["filter_exclude"] = new_exc
        _save_config_yaml(config_yaml)
        st.success("Filters saved to config.yaml.")

    st.markdown("---")

    # -- Job Board API Keys --
    st.subheader("Job Board API Keys")
    st.caption("Optional — configure to search additional job boards beyond Indeed and LinkedIn.")

    config_yaml = _load_config_yaml()
    usa_cfg = config_yaml.get("usajobs", {})
    adz_cfg = config_yaml.get("adzuna", {})
    joo_cfg = config_yaml.get("jooble", {})

    usa_key = usa_cfg.get("api_key", "") or os.environ.get("USAJOBS_API_KEY", "")
    usa_email = usa_cfg.get("email", "") or os.environ.get("USAJOBS_EMAIL", "")
    adz_id = adz_cfg.get("app_id", "") or os.environ.get("ADZUNA_APP_ID", "")
    adz_key = adz_cfg.get("app_key", "") or os.environ.get("ADZUNA_APP_KEY", "")
    joo_key = joo_cfg.get("api_key", "") or os.environ.get("JOOBLE_API_KEY", "")

    col_usa1, col_usa2 = st.columns(2)
    with col_usa1:
        new_usa_key = st.text_input("USAJobs API Key", value=usa_key, type="password", key="setup_key_usajobs")
    with col_usa2:
        new_usa_email = st.text_input("USAJobs Email", value=usa_email, key="setup_email_usajobs")
    col_adz1, col_adz2 = st.columns(2)
    with col_adz1:
        new_adz_id = st.text_input("Adzuna App ID", value=adz_id, type="password", key="setup_key_adzuna_id")
    with col_adz2:
        new_adz_key = st.text_input("Adzuna App Key", value=adz_key, type="password", key="setup_key_adzuna_key")
    new_joo_key = st.text_input("Jooble API Key", value=joo_key, type="password", key="setup_key_jooble")

    if st.button("Save Job Board API Keys", key="setup_save_keys"):
        config_yaml = _load_config_yaml()
        config_yaml.setdefault("usajobs", {})
        config_yaml.setdefault("adzuna", {})
        config_yaml.setdefault("jooble", {})

        if new_usa_key:
            config_yaml["usajobs"]["api_key"] = new_usa_key
        if new_usa_email:
            config_yaml["usajobs"]["email"] = new_usa_email
        if new_adz_id:
            config_yaml["adzuna"]["app_id"] = new_adz_id
        if new_adz_key:
            config_yaml["adzuna"]["app_key"] = new_adz_key
        if new_joo_key:
            config_yaml["jooble"]["api_key"] = new_joo_key

        _save_config_yaml(config_yaml)
        st.success("Job board API keys saved to config.yaml.")

    # ===================================================================
    # Group 4: Evaluator Patterns
    # ===================================================================
    st.header("Evaluator Patterns")

    patterns = _load_evaluator_patterns()

    if patterns:
        pattern_keys = ["skip_title_patterns", "skip_description_patterns",
                        "rescue_patterns", "boost_patterns"]
        tab_labels = ["Skip Title", "Skip Description", "Rescue", "Boost"]
        tab_descriptions = {
            "skip_title_patterns": "Jobs with these words in the title are automatically skipped",
            "skip_description_patterns": "Jobs with these phrases in the description are skipped, unless rescued",
            "rescue_patterns": "Core skills that save a job from being skipped",
            "boost_patterns": "Dream-job keywords that get priority evaluation",
        }
        tabs_d = st.tabs(tab_labels)

        edited_dfs: dict[str, pd.DataFrame] = {}
        for tab, key in zip(tabs_d, pattern_keys):
            with tab:
                st.caption(tab_descriptions[key])
                pats = patterns.get(key, [])
                # Convert regex patterns to display strings
                rows = []
                for pat in pats:
                    display, _can_rt = regex_to_display(pat)
                    rows.append({"Pattern": display})
                if not rows:
                    rows.append({"Pattern": ""})
                df = pd.DataFrame(rows)
                edited = st.data_editor(
                    df,
                    column_config={
                        "Pattern": st.column_config.TextColumn(
                            "Pattern",
                            help="Enter a plain-English phrase (e.g. 'drug substance') "
                                 "or raw regex (e.g. '\\bstats\\b')",
                            width="large",
                        ),
                    },
                    num_rows="dynamic",
                    use_container_width=True,
                    key=f"setup_pat_{key}",
                )
                edited_dfs[key] = edited

        if st.button("Save Patterns", key="setup_save_patterns"):
            new_patterns = {}
            errors = []
            for key in pattern_keys:
                df = edited_dfs[key]
                valid = []
                for _, row in df.iterrows():
                    pat_str = str(row.get("Pattern", "")).strip()
                    if not pat_str:
                        continue
                    # display_to_regex handles both: phrases get converted,
                    # raw regex (containing \b or \s) passes through as-is
                    regex_str = display_to_regex(pat_str)
                    # Validate
                    try:
                        re.compile(regex_str)
                        valid.append(regex_str)
                    except re.error as e:
                        errors.append(f"Invalid regex skipped ({pat_str}): {e}")
                new_patterns[key] = valid
            _save_evaluator_patterns(new_patterns)
            # Reload evaluator patterns if evaluator is imported
            try:
                from src.evaluator import reload_patterns
                reload_patterns()
            except Exception:
                pass
            if errors:
                for err in errors:
                    st.warning(err)
            st.success("Evaluator patterns saved.")

        # Regex test tool in expander
        with st.expander("Advanced: Regex Test Tool"):
            test_text = st.text_input("Test text:", key="setup_test_pattern_text")
            test_regex = st.text_input("Regex pattern:", key="setup_test_pattern_regex")
            if test_text and test_regex:
                try:
                    match = re.search(test_regex, test_text, re.IGNORECASE)
                    if match:
                        st.success(f"Match found: '{match.group()}'")
                    else:
                        st.warning("No match.")
                except re.error as e:
                    st.error(f"Invalid regex: {e}")
    else:
        st.info("No evaluator patterns file found. Run the setup wizard to generate one, "
                "or the built-in patterns will be used automatically.")

    # ===================================================================
    # Group 5: Advanced (collapsed)
    # ===================================================================
    st.header("Advanced")

    with st.expander("Resume Profile (JSON)"):
        profile = _load_resume_profile()

        if profile:
            profile_text = json.dumps(profile, indent=2)
            st.caption("Edit your resume profile (JSON format) and click Save.")
            edited_profile = st.text_area(
                "resume_profile.json", value=profile_text, height=400, key="setup_profile_editor"
            )
            if st.button("Save Resume Profile", key="setup_save_profile"):
                try:
                    parsed = json.loads(edited_profile)
                    _save_resume_profile(parsed)
                    st.success("Resume profile saved.")
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")
        else:
            st.info("No resume profile found. Use the Setup Wizard above to generate one, "
                    "or copy `data/resume_profile.example.json` to `data/resume_profile.json`.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("Pharma/Biotech Job Search")

    # --- Sidebar: Search & Reload ---
    st.sidebar.header("Actions")

    search_days = st.sidebar.number_input(
        "Search lookback (days)", min_value=1, max_value=90, value=7,
        help="How many days back to search for jobs",
    )

    if st.sidebar.button("\U0001f50d Run New Search", use_container_width=True, type="primary"):
        run_search_from_dashboard(search_days=search_days)
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")

    eval_days = st.sidebar.number_input(
        "Evaluate jobs from last N days", min_value=1, max_value=90, value=1,
        help="Only evaluate jobs posted within this many days",
    )

    if st.sidebar.button("\U0001f9ea Evaluate Jobs", use_container_width=True, type="primary"):
        run_evaluation_from_dashboard(eval_days=eval_days)
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")

    if st.sidebar.button("\U0001f504 Reload Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")

    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs([
        "\U0001f4cb  Job Listings",
        "\U0001f4ca  Evaluation Results",
        "\u2699\ufe0f  Setup",
    ])

    # Setup tab renders independently (no data dependency)
    with tab3:
        render_setup_tab()

    # Data tabs share loaded data
    df = load_data()

    if df.empty:
        with tab1:
            st.info("No data yet. Click **Run New Search** in the sidebar, "
                    "or run from CLI: `python job_search.py`")
        with tab2:
            st.info("No evaluation data yet. Run a search first, then evaluate jobs.")
        return

    # Compute derived columns
    if "date_posted" in df.columns:
        df["date_posted"] = pd.to_datetime(df["date_posted"], errors="coerce")
        today = pd.Timestamp.now().normalize()
        df["days_since_posted"] = (today - df["date_posted"]).dt.days
        df["date_posted"] = df["date_posted"].dt.strftime("%Y-%m-%d")

    if "reposted_date" not in df.columns:
        df["reposted_date"] = ""
    df["reposted_date"] = df["reposted_date"].fillna("")

    # Ensure eval_status column exists (default "" = unevaluated/pending)
    if "eval_status" not in df.columns:
        df["eval_status"] = ""
    df["eval_status"] = df["eval_status"].fillna("")

    # Backfill eval_status from evaluations.json for jobs evaluated before this column existed
    config = build_config()
    eval_path = PROJECT_ROOT / config.evaluation.evaluations_store
    if eval_path.exists():
        try:
            with open(eval_path, "r") as f:
                eval_data = json.load(f)
            evals = eval_data.get("evaluations", {})
            if evals:
                needs_backfill = df[(df["eval_status"] == "") & (df["job_url"].isin(evals))]
                if not needs_backfill.empty:
                    for idx, row in needs_backfill.iterrows():
                        ev = evals.get(row["job_url"], {})
                        bucket = ev.get("fit_bucket", "")
                        if bucket == "prefilter_skip":
                            df.at[idx, "eval_status"] = "skipped"
                        elif bucket:
                            df.at[idx, "eval_status"] = "evaluated"
        except (json.JSONDecodeError, IOError):
            pass

    df["job_code"] = df.apply(
        lambda row: extract_job_code(row.get("job_url", ""), row.get("source", "")),
        axis=1,
    )

    # Load reviewed data (shared across both tabs)
    reviewed_data = load_reviewed()
    df["reviewed_at"] = df["job_url"].map(reviewed_data).fillna("")

    with tab1:
        render_job_listings_tab(df.copy(), reviewed_data)

    with tab2:
        render_evaluation_tab(df.copy(), reviewed_data)


if __name__ == "__main__":
    main()
