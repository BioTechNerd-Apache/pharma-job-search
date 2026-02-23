"""Streamlit web dashboard for browsing job search results and evaluation results."""

import json
import os
import re
import sys
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

from src.config import build_config, PROJECT_ROOT
from src.exporter import get_master_path

st.set_page_config(
    page_title="Pharma/Biotech Job Search",
    page_icon="\U0001f52c",
    layout="wide",
)

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


def run_search_from_dashboard():
    """Run a new search from the dashboard with progress tracking."""
    from src.aggregator import run_search

    config = build_config()

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
    elif "jl_selected" not in st.session_state:
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

    min_score = st.sidebar.slider("Min Fit Score", 0, 100, 0, key="eval_min_score")
    if min_score > 0:
        eval_df = eval_df[eval_df["fit_score"] >= min_score]

    if "recommendation" in eval_df.columns:
        recs = sorted(eval_df["recommendation"].dropna().unique())
        selected_recs = st.sidebar.multiselect("Recommendation", recs, default=recs, key="eval_recs")
        if selected_recs:
            eval_df = eval_df[eval_df["recommendation"].isin(selected_recs)]

    if "fit_bucket" in eval_df.columns:
        buckets = sorted(eval_df["fit_bucket"].dropna().unique())
        selected_buckets = st.sidebar.multiselect("Fit Bucket", buckets, default=buckets, key="eval_buckets")
        if selected_buckets:
            eval_df = eval_df[eval_df["fit_bucket"].isin(selected_buckets)]

    reviewed_count = (eval_df["reviewed_at"] != "").sum()
    unreviewed_count = len(eval_df) - reviewed_count
    show_unreviewed = st.sidebar.checkbox(
        f"Unreviewed only ({unreviewed_count} remaining)", value=False, key="eval_unreviewed"
    )
    if show_unreviewed:
        eval_df = eval_df[eval_df["reviewed_at"] == ""]

    if "source" in eval_df.columns:
        sources = sorted(eval_df["source"].dropna().unique())
        selected_sources = st.sidebar.multiselect("Source", sources, default=sources, key="eval_sources")
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
        grid_df = grid_df.sort_values(sort_cols, ascending=[False, True]).reset_index(drop=True)

    gb = GridOptionsBuilder.from_dataframe(grid_df)
    gb.configure_default_column(filterable=True, sortable=True, resizable=True, filter=True)

    # Fit score with color coding
    if "fit_score" in display_cols:
        gb.configure_column("fit_score", headerName="Fit Score",
                            filter="agNumberColumnFilter", cellStyle=FIT_SCORE_STYLE)

    if "fit_bucket" in display_cols:
        gb.configure_column("fit_bucket", headerName="Fit Bucket", filter="agSetColumnFilter")

    if "recommendation" in display_cols:
        gb.configure_column("recommendation", headerName="Recommendation", filter="agSetColumnFilter")

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
            gb.configure_column(col, headerName="Days Old", filter="agNumberColumnFilter")

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
    elif "eval_selected" not in st.session_state:
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
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("Pharma/Biotech Job Search Results")

    # --- Sidebar: Search & Reload ---
    st.sidebar.header("Actions")

    if st.sidebar.button("\U0001f50d Run New Search", use_container_width=True, type="primary"):
        run_search_from_dashboard()
        st.cache_data.clear()
        st.rerun()

    if st.sidebar.button("\U0001f504 Reload Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown("---")

    df = load_data()

    if df.empty:
        st.warning("No data found. Click **Run New Search** or run from CLI: `python job_search.py`")
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

    # --- Tabs ---
    tab1, tab2 = st.tabs(["Job Listings", "Evaluation Results"])

    with tab1:
        render_job_listings_tab(df.copy(), reviewed_data)

    with tab2:
        render_evaluation_tab(df.copy(), reviewed_data)


if __name__ == "__main__":
    main()
