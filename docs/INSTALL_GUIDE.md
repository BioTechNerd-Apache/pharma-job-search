# Installation Guide

A step-by-step guide for installing the Pharma/Biotech Job Search Tool on **Windows** or **Mac**. No terminal experience required.

---

## Table of Contents

1. [Prerequisites Checklist](#1-prerequisites-checklist)
2. [Installing Prerequisites](#2-installing-prerequisites)
3. [Install the Tool](#3-install-the-tool)
4. [Configure the Tool](#4-configure-the-tool)
5. [Run Your First Search](#5-run-your-first-search)
6. [Create a Desktop Shortcut (Optional)](#6-create-a-desktop-shortcut-optional)
7. [Quick Reference Cheat Sheet](#7-quick-reference-cheat-sheet)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites Checklist

You need three things installed before you start:

| Prerequisite | Required Version | Download Link |
|---|---|---|
| **Python** | 3.10, 3.11, 3.12, or 3.13 | [python.org/downloads](https://www.python.org/downloads/) |
| **pip** | Any (comes with Python) | Included with Python |
| **Git** | Any | [git-scm.com/downloads](https://git-scm.com/downloads) |

> **Warning**: Do **NOT** install Python 3.14 or any version marked "pre-release" or "alpha/beta". Many dependencies (like NumPy and pandas) will fail to install on pre-release Python. Stick to **3.10, 3.11, 3.12, or 3.13**.

### Already installed? Verify with these commands

Open a terminal (Mac: **Terminal** app; Windows: **Command Prompt** or **PowerShell**) and run:

**Check Python:**
```
python --version
```
Expected output (any of these is fine):
```
Python 3.12.8
```

**Check pip:**
```
pip --version
```
Expected output:
```
pip 24.3.1 from /usr/lib/python3.12/site-packages/pip (python 3.12)
```

**Check Git:**
```
git --version
```
Expected output:
```
git version 2.43.0
```

If any of these commands show an error like `'python' is not recognized` or `command not found`, follow the installation steps in the next section.

> **Windows note**: If `python` doesn't work, try `py` instead. Windows sometimes uses `py` as the Python command.

---

## 2. Installing Prerequisites

### Windows

#### Install Python

1. Go to [python.org/downloads](https://www.python.org/downloads/)
2. Download the latest **stable** release (look for 3.12.x or 3.13.x — do NOT pick anything labeled "pre-release")
3. Run the installer
4. **IMPORTANT**: On the very first screen of the installer, check the box that says **"Add python.exe to PATH"** — this is easy to miss and causes most install problems

   ![Add to PATH checkbox](https://docs.python.org/3/_images/win_installer.png)

5. Click "Install Now"
6. When it finishes, click "Disable path length limit" if prompted, then close the installer

#### Install Git

1. Go to [git-scm.com/downloads/win](https://git-scm.com/downloads/win)
2. Download the installer
3. Run it and click "Next" through all the screens (the defaults are fine)
4. Click "Install", then "Finish"

#### Verify the installations

**Close and reopen** your Command Prompt or PowerShell (this is important — the new PATH settings only take effect in new windows), then run:

```
python --version
pip --version
git --version
```

All three commands should print version numbers, not errors. If `python` doesn't work, try `py --version` instead.

---

### Mac

#### Install Python

**Option A: Using Homebrew (recommended if you have Homebrew)**

If you already have Homebrew installed, run:
```bash
brew install python@3.12
```

**Option B: Using the installer from python.org**

1. Go to [python.org/downloads](https://www.python.org/downloads/)
2. Download the macOS installer for the latest stable 3.12.x or 3.13.x
3. Open the `.pkg` file and follow the prompts
4. When the install finishes, it may open a Finder window — you can close it

> **Tip**: On Mac, you may need to use `python3` and `pip3` instead of `python` and `pip`. This guide will use `python` for simplicity, but substitute `python3`/`pip3` if needed.

#### Install Git

Git is usually already installed on Mac. To check, run:
```bash
git --version
```

If you see a version number, you're all set. If you get a prompt to install "command line developer tools", click **Install** and wait for it to finish. You can also install manually:
```bash
xcode-select --install
```

#### Verify the installations

Open a **new** Terminal window and run:
```bash
python3 --version
pip3 --version
git --version
```

All three should print version numbers.

---

## 3. Install the Tool

Pick **one** of these two methods:

### Method A: One-command install (recommended)

This installs the tool and all its dependencies in one step:

**Mac:**
```bash
pip3 install git+https://github.com/BioTechNerd-Apache/pharma-job-search.git
```

**Windows:**
```
pip install git+https://github.com/BioTechNerd-Apache/pharma-job-search.git
```

This creates a `pharma-job-search` command you can run from anywhere.

### Method B: Clone and install

If you want a local copy of the code (useful if you plan to customize):

**Mac:**
```bash
git clone https://github.com/BioTechNerd-Apache/pharma-job-search.git
cd pharma-job-search
pip3 install -r requirements.txt
```

**Windows:**
```
git clone https://github.com/BioTechNerd-Apache/pharma-job-search.git
cd pharma-job-search
pip install -r requirements.txt
```

### Verify the install

**If you used Method A:**
```
pharma-job-search --help
```

**If you used Method B:**
```
python job_search.py --help
```

You should see a help message listing available commands. If you see an error, check the [Troubleshooting](#8-troubleshooting) section.

---

## 4. Configure the Tool

Everything is configured through the **dashboard's Setup tab** — no need to edit config files by hand.

> **Note**: `config.yaml` is auto-created the first time you launch the dashboard. You do **not** need to copy it manually.

### Launch the dashboard

**If you used Method A (pip install):**
```
pharma-job-search --web
```

**If you used Method B (clone):**
```
python job_search.py --web
```

Your browser will open to `http://localhost:8501`. Click the **Setup / Profile** tab at the top.

### Walk through the Setup tab

1. **Choose an AI provider** — Pick Anthropic, OpenAI, or Ollama (free) and enter your API key. Ollama requires no key — just install it from [ollama.com/download](https://ollama.com/download) and run `ollama pull llama3.1:8b` in your terminal first.
2. **Upload your resume** — Click "Upload Resume" and select your PDF, DOCX, or TXT file, then click **Run Setup Wizard**. The AI reads your resume and generates search terms, filters, evaluator patterns, and a resume profile — all automatically.
3. **Review and edit** — After the wizard runs, you can tweak search terms, filters, API keys, and patterns in the same Setup tab. All changes auto-save to config files.

That's it — everything is configured. You can close the Setup tab and start searching.

### AI provider options

| Provider | Cost | Setup |
|----------|------|-------|
| **Ollama (free)** | Free | Install from [ollama.com/download](https://ollama.com/download), then run `ollama pull llama3.1:8b` |
| **Anthropic (Claude)** | Paid | Get key at [console.anthropic.com](https://console.anthropic.com/settings/keys) |
| **OpenAI (GPT)** | Paid | Get key at [platform.openai.com](https://platform.openai.com/api-keys) |

### Optional job board API keys

None of these are required — Indeed and LinkedIn work without any keys. You can add these in the Setup tab under "Job Board API Keys" if you want more sources.

| API Key | What it adds | Where to get it |
|---|---|---|
| USAJobs | Federal job listings | [developer.usajobs.gov](https://developer.usajobs.gov/) |
| Adzuna | Adzuna job listings | [developer.adzuna.com](https://developer.adzuna.com/) |
| Jooble | Jooble job listings | [jooble.org/api/about](https://jooble.org/api/about) |

### Terminal alternative (optional)

If you prefer to set up entirely from the terminal, you can run the setup wizard as a CLI command. You don't need to edit any files by hand — the wizard is interactive and prompts you for everything.

**What you need before running the wizard:**
- Your resume file (PDF, DOCX, or TXT) saved somewhere on your computer
- **One** of these AI providers ready:
  - **Ollama** (free) — install from [ollama.com/download](https://ollama.com/download), then run `ollama pull llama3.1:8b`
  - **Anthropic** — have your API key from [console.anthropic.com](https://console.anthropic.com/settings/keys)
  - **OpenAI** — have your API key from [platform.openai.com](https://platform.openai.com/api-keys)

**Run the wizard:**

```
python job_search.py --setup resume.pdf
```

Replace `resume.pdf` with the actual path to your resume file.

**What happens step by step:**

1. **API keys** — The wizard prompts you for each API key. Type or paste the key and press Enter, or just press Enter to skip optional ones:
   ```
   Enter Anthropic API key (or press Enter to skip): sk-ant-...
   Enter USAJobs API key (or press Enter to skip): [Enter]
   Enter Adzuna App ID (or press Enter to skip): [Enter]
   Enter Jooble API key (or press Enter to skip): [Enter]
   ```
   > If you're using Ollama, you don't need any API key — just make sure Ollama is running (`ollama serve`).

2. **Resume profile** — The AI reads your resume and generates a structured profile (career history, skills, target domains). You'll see a preview and be asked:
   ```
   Accept this resume profile? [Y/n]:
   ```
   Type `Y` (or just press Enter) to accept, or `n` to skip.

3. **Search configuration** — The AI generates search terms, priority terms, synonyms, and include/exclude filters based on your resume. You'll see them listed and be asked to confirm.

4. **Evaluator patterns** — The AI generates regex patterns for the pre-filter (what to skip, rescue, and boost). Confirm again.

5. **Save** — The wizard shows which files it will create and asks for final confirmation:
   ```
   Files to write:
     - config.yaml
     - data/resume_profile.json
     - data/evaluator_patterns.yaml
   Proceed with saving? [Y/n]:
   ```

That's it. The wizard creates `config.yaml` and all data files — you don't need to manually copy or edit anything. You can always fine-tune the generated config later in the dashboard's Setup tab.

> **Note**: You do **not** need to copy `config.yaml` first. The wizard creates it automatically. If one already exists, your existing settings are preserved and only new values are added.

---

## 5. Run Your First Search

### From the dashboard (recommended)

If you still have the dashboard open from the previous step, click **Run New Search** in the left sidebar. This scrapes all configured job boards and displays the results in the Job Listings tab.

If you closed the dashboard, launch it again:

**If you used Method A:**
```
pharma-job-search --web
```

**If you used Method B:**
```
python job_search.py --web
```

### From the terminal (alternative)

```
python job_search.py --days 1
```

This searches for jobs posted in the last 24 hours. Results are saved to `data/pharma_jobs.csv` and `data/pharma_jobs.xlsx`. Open the dashboard afterwards to browse them.

### Evaluate jobs with AI (optional)

From the terminal:
```
python job_search.py --evaluate
```

This searches for jobs and then scores them against your resume using AI. Requires an AI provider to be configured (Anthropic, OpenAI, or Ollama).

---

## 6. Create a Desktop Shortcut (Optional)

Create a shortcut on your Desktop that launches the dashboard with one click:

**If you used Method A (pip install):**
```
pharma-job-search --create-shortcut
```

**If you used Method B (clone):**
```
python job_search.py --create-shortcut
```

This creates:
- **Mac**: `~/Desktop/Pharma Job Search.command` (double-click to launch)
- **Windows**: `~/Desktop/Pharma Job Search.bat` (double-click to launch)

---

## 7. Quick Reference Cheat Sheet

### GUI path (recommended) — install, launch, done

**Mac:**
```bash
brew install python@3.12                    # 1. Install Python (skip if already installed)
pip3 install git+https://github.com/BioTechNerd-Apache/pharma-job-search.git  # 2. Install the tool
pharma-job-search --web                     # 3. Launch dashboard — configure in Setup tab
```

**Windows:**
```
REM 1. Install Python from https://www.python.org/downloads/ (check "Add to PATH")
REM 2. Install Git from https://git-scm.com/downloads/win
REM 3. Close and reopen Command Prompt, then:
pip install git+https://github.com/BioTechNerd-Apache/pharma-job-search.git
pharma-job-search --web
```

### Terminal path (alternative) — install, wizard, search

**Mac:**
```bash
brew install python@3.12                    # 1. Install Python (skip if already installed)
pip3 install git+https://github.com/BioTechNerd-Apache/pharma-job-search.git  # 2. Install the tool
pharma-job-search --setup resume.pdf        # 3. Interactive wizard (prompts for API keys)
pharma-job-search --web                     # 4. Launch dashboard to browse results
```

**Windows:**
```
REM 1. Install Python from https://www.python.org/downloads/ (check "Add to PATH")
REM 2. Install Git from https://git-scm.com/downloads/win
REM 3. Close and reopen Command Prompt, then:
pip install git+https://github.com/BioTechNerd-Apache/pharma-job-search.git
pharma-job-search --setup resume.pdf
pharma-job-search --web
```

---

## 8. Troubleshooting

### "pip is not recognized" or "pip: command not found"

**Cause**: pip is not on your system PATH.

**Fix**: Use `python -m pip` instead of `pip`:
```
python -m pip install git+https://github.com/BioTechNerd-Apache/pharma-job-search.git
```

On Mac, use `python3 -m pip` instead of `pip3`.

If that also fails, reinstall Python and make sure to check **"Add python.exe to PATH"** (Windows) during installation.

---

### "git is not recognized" or "git: command not found"

**Cause**: Git is not installed or not on your PATH.

**Fix**:
- **Windows**: Download and install Git from [git-scm.com/downloads/win](https://git-scm.com/downloads/win), then **close and reopen** your Command Prompt
- **Mac**: Run `xcode-select --install` to install the command line tools

---

### "python is not recognized" (Windows)

**Cause**: Python is not on your PATH.

**Fixes** (try in order):
1. Try `py` instead of `python` (e.g., `py job_search.py --web`)
2. Try `py -m pip` instead of `pip`
3. Reinstall Python — make sure to check the **"Add python.exe to PATH"** checkbox on the first screen of the installer
4. After reinstalling, **close and reopen** Command Prompt

---

### NumPy, pandas, or other packages fail to install with compile errors

**Cause**: You likely have Python 3.14 (pre-release) installed. Pre-release Python versions don't have pre-built packages (called "wheels") for most libraries, so pip tries to compile them from source — which usually fails.

**Fix**:
1. Uninstall Python 3.14
2. Install Python **3.12** or **3.13** (the latest stable versions) from [python.org/downloads](https://www.python.org/downloads/)
3. Make sure to check **"Add python.exe to PATH"** during installation
4. Close and reopen your terminal
5. Verify: `python --version` should show 3.12.x or 3.13.x
6. Try the install again

---

### "python3: command not found" (Mac)

**Cause**: Python 3 isn't installed or Homebrew isn't linked.

**Fix**:
```bash
brew install python@3.12
```

Or download the installer from [python.org/downloads](https://www.python.org/downloads/).

---

### PowerShell says "running scripts is disabled on this system"

**Cause**: PowerShell's execution policy blocks scripts.

**Fix**: Use **Command Prompt** instead of PowerShell. Search for "cmd" in the Start menu and open "Command Prompt". All the commands in this guide work in Command Prompt.

Alternatively, open PowerShell as Administrator and run:
```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### "Permission denied" or "ERROR: Could not install packages" (Mac)

**Cause**: You don't have permission to install to the system Python directory.

**Fixes** (try in order):
1. Add `--user` to install to your user directory:
   ```bash
   pip3 install --user git+https://github.com/BioTechNerd-Apache/pharma-job-search.git
   ```
2. Or use a virtual environment (recommended):
   ```bash
   python3 -m venv ~/pharma-env
   source ~/pharma-env/bin/activate
   pip install git+https://github.com/BioTechNerd-Apache/pharma-job-search.git
   ```
   You'll need to run `source ~/pharma-env/bin/activate` each time you open a new terminal before using the tool.

---

### "No module named 'streamlit'" or similar import errors

**Cause**: Dependencies didn't install correctly.

**Fix**: Reinstall the requirements:
```
pip install -r requirements.txt
```

Or if you used Method A:
```
pip install --force-reinstall git+https://github.com/BioTechNerd-Apache/pharma-job-search.git
```

---

### The dashboard doesn't open in my browser

**Cause**: The browser didn't launch automatically.

**Fix**: After running `python job_search.py --web`, manually open your browser and go to:
```
http://localhost:8501
```

---

### Still stuck?

Open an issue at [github.com/BioTechNerd-Apache/pharma-job-search/issues](https://github.com/BioTechNerd-Apache/pharma-job-search/issues) with:
- Your operating system (Windows/Mac) and version
- The exact command you ran
- The full error message you see
