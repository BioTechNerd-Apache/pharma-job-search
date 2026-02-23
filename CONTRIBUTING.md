# Contributing

Thanks for your interest in contributing to the Pharma/Biotech Job Search Tool!

## Getting Started

### Dev Environment Setup

```bash
git clone https://github.com/BioTechNerd-Apache/pharma-job-search.git
cd pharma-job-search
pip install -r requirements.txt
cp config.example.yaml config.yaml
cp data/resume_profile.example.json data/resume_profile.json
```

Edit `config.yaml` with your API keys (Indeed and LinkedIn work without any keys).

### Running Tests

```bash
python -m pytest tests/
```

### Running the Dashboard

```bash
python job_search.py --web
```

## Customizing for Your Discipline

This tool ships pre-configured for pharma/biotech scientist roles, but it's designed to be adapted. There are 4 layers to customize:

1. **Search terms** — `config.yaml` → `search.terms` + `search.synonyms`
2. **Discipline filters** — `config.yaml` → `search.filter_include` + `search.filter_exclude`
3. **Pre-filter patterns** — `src/evaluator.py` → skip/rescue/boost pattern lists
4. **Resume profile** — `data/resume_profile.json`

See the [Customization Guide](README.md#customization-guide) in the README for full details.

## Reporting Bugs

Open a [Bug Report](https://github.com/BioTechNerd-Apache/pharma-job-search/issues/new?template=bug_report.md) with:
- Steps to reproduce
- Error output / traceback
- Your OS and Python version
- Which job sources you're using

## Suggesting Features

Open a [Feature Request](https://github.com/BioTechNerd-Apache/pharma-job-search/issues/new?template=feature_request.md) describing the problem you want to solve and your proposed approach.

## Code Style

- Python 3.10+ with type hints where practical
- Functions and modules have docstrings for public APIs
- Scrapers return pandas DataFrames with the standard column schema (see `src/aggregator.py:OUTPUT_COLUMNS`)
- Config values go in `config.yaml`, not hardcoded
- Keep CLI flags consistent with the existing `--eval-*` pattern

## Pull Requests

1. Fork the repo and create a feature branch
2. Make your changes
3. Run tests: `python -m pytest tests/`
4. Test the dashboard: `python job_search.py --web`
5. Open a PR with a clear description of what changed and why
