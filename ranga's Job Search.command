#!/usr/bin/env bash
# ============================================================
# Ranga's Job Search — Dashboard Only (no scraping)
# Double-click to view your latest job search results in Chrome
# ============================================================

PROJECT_DIR="$HOME/Library/Mobile Documents/com~apple~CloudDocs/My Vibe Code/pharma-job-search"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
PORT=8501
URL="http://localhost:$PORT"

# Kill any existing Streamlit on this port
lsof -ti :$PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
sleep 1

echo "========================================"
echo "  Ranga's Job Search Dashboard"
echo "========================================"
echo ""
echo "Loading latest data from: $PROJECT_DIR/data/"
echo ""

# Force iCloud to download data files before launching dashboard
# (prevents stale data when opening on a different Mac)
DATA_DIR="$PROJECT_DIR/data"
if [ -d "$DATA_DIR" ]; then
    echo "Syncing data files from iCloud..."
    for f in "$DATA_DIR"/pharma_jobs.csv "$DATA_DIR"/pharma_jobs.xlsx "$DATA_DIR"/pharma_jobs_raw.csv "$DATA_DIR"/reviewed.json; do
        if [ -e "$f" ]; then
            brctl download "$f" 2>/dev/null
        fi
    done
    # Wait up to 30 seconds for the main CSV to finish downloading
    for i in $(seq 1 30); do
        if [ -f "$DATA_DIR/pharma_jobs.csv" ] && ! brctl download "$DATA_DIR/pharma_jobs.csv" 2>&1 | grep -q "not evicted"; then
            # File is local — verify it's not still being written by checking stable size
            SIZE1=$(stat -f%z "$DATA_DIR/pharma_jobs.csv" 2>/dev/null || echo 0)
            sleep 1
            SIZE2=$(stat -f%z "$DATA_DIR/pharma_jobs.csv" 2>/dev/null || echo 0)
            if [ "$SIZE1" = "$SIZE2" ] && [ "$SIZE1" -gt 0 ]; then
                echo "Data files synced."
                break
            fi
        fi
        echo "  Waiting for iCloud sync... ($i/30)"
        sleep 1
    done
fi

# Start Streamlit dashboard only (NO scraping)
cd "$PROJECT_DIR"
"$PYTHON" -m streamlit run src/dashboard.py \
    --server.port $PORT \
    --server.headless true \
    --browser.gatherUsageStats false &

STREAMLIT_PID=$!

# Wait for server to be ready
echo "Starting dashboard..."
for i in $(seq 1 30); do
    if curl -s "$URL" > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

# Open in Chrome
if [ -d "/Applications/Google Chrome.app" ]; then
    open -a "Google Chrome" "$URL"
else
    open "$URL"
fi

echo ""
echo "Dashboard running at: $URL"
echo ""
echo "Click the '🔍 Run New Search' button in the sidebar to search for new jobs."
echo "Or run from Terminal:  cd \"$PROJECT_DIR\" && python3 job_search.py"
echo ""
echo "Press Ctrl+C to stop the dashboard."

wait $STREAMLIT_PID
