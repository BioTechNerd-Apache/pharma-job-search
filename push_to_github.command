#!/usr/bin/env bash
# ============================================================
# Pharma Job Search — Push to GitHub
# Double-click to commit and push local changes to GitHub
# API keys and data files are protected by .gitignore
# ============================================================

PROJECT_DIR="$HOME/Library/Mobile Documents/com~apple~CloudDocs/My Vibe Code/pharma-job-search"

cd "$PROJECT_DIR" || { echo "ERROR: Project directory not found!"; read -rp "Press Enter to close..."; exit 1; }

echo "========================================"
echo "  Push to GitHub"
echo "========================================"
echo ""

# Check if we're in a git repo
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "ERROR: Not a git repository!"
    read -rp "Press Enter to close..."
    exit 1
fi

# Show current branch
BRANCH=$(git branch --show-current)
echo "Branch: $BRANCH"
echo "Remote: $(git remote get-url origin 2>/dev/null)"
echo ""

# Check for changes
if git diff --quiet HEAD && [ -z "$(git ls-files --others --exclude-standard)" ]; then
    echo "No changes to commit. Everything is up to date."
    echo ""
    read -rp "Press Enter to close..."
    exit 0
fi

# Show what's changed
echo "--- Changed/modified files ---"
git diff --name-status HEAD 2>/dev/null
echo ""
echo "--- New untracked files ---"
git ls-files --others --exclude-standard 2>/dev/null
echo ""

# Safety check: verify .gitignore is protecting sensitive files
echo "--- Protected by .gitignore (will NOT be pushed) ---"
for f in config.yaml data/pharma_jobs.csv data/pharma_jobs.xlsx data/evaluations.json data/resume_profile.json data/reviewed.json; do
    if [ -e "$f" ]; then
        echo "  $f"
    fi
done
echo ""

# Ask for commit message
DEFAULT_MSG="Update pharma-job-search ($(date '+%Y-%m-%d %H:%M'))"
echo "Enter a commit message (or press Enter for default):"
echo "  Default: \"$DEFAULT_MSG\""
echo ""
read -rp "> " CUSTOM_MSG

if [ -z "$CUSTOM_MSG" ]; then
    COMMIT_MSG="$DEFAULT_MSG"
else
    COMMIT_MSG="$CUSTOM_MSG"
fi

echo ""
echo "Staging files..."
git add -A

echo "Committing..."
git commit -m "$COMMIT_MSG"

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Commit failed! See message above."
    read -rp "Press Enter to close..."
    exit 1
fi

echo ""
echo "Pushing to GitHub..."
git push origin "$BRANCH" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "  Successfully pushed to GitHub!"
    echo "========================================"
    echo ""
    git log --oneline -1
else
    echo ""
    echo "ERROR: Push failed! Check your network connection or GitHub credentials."
fi

echo ""
read -rp "Press Enter to close..."
