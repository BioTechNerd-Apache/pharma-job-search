"""Tests for src.pattern_helpers — regex ↔ display round-trip conversions."""

import re

import pytest
import yaml

from src.pattern_helpers import display_to_regex, regex_to_display


# ---------------------------------------------------------------------------
# Round-trip: regex → display → regex  (must reproduce the original)
# ---------------------------------------------------------------------------

# Real patterns from data/evaluator_patterns.yaml and evaluator builtins
ROUNDTRIP_CASES = [
    # Single word
    (r"\bprocurement\b", "procurement"),
    (r"\bvp\b", "vp"),
    (r"\bcrispr\b", "crispr"),
    (r"\belisa\b", "elisa"),
    (r"\bimmunology\b", "immunology"),
    # Hyphenated word
    (r"\blc-ms\b", "lc-ms"),
    (r"\brt-pcr\b", "rt-pcr"),
    # Two-word phrase (\s+ connector)
    (r"\bdrug\s+substance\b", "drug substance"),
    (r"\bgeneral\s+counsel\b", "general counsel"),
    (r"\bdata\s+scientist\b", "data scientist"),
    (r"\bentry\s+level\b", "entry level"),
    (r"\bvaccine\s+development\b", "vaccine development"),
    (r"\bin\s+vivo\s+pharmacology\b", "in vivo pharmacology"),
    (r"\bin\s+vivo\s+toxicology\b", "in vivo toxicology"),
    # Three-word phrase
    (r"\bnext\s+generation\s+sequencing\b", "next generation sequencing"),
    (r"\badministrative\s+assistant\b", "administrative assistant"),
    (r"\bregulatory\s+affairs\s+officer\b", "regulatory affairs officer"),
    # Wildcard gap (.*  between \b-bounded words)
    (r"\bchief\b.*\bofficer\b", "chief ... officer"),
    (r"\bexecutive\b.*\bvice\b", "executive ... vice"),
    (r"\bmd\b.*\bdirector\b", "md ... director"),
    (r"\bfinance\b.*\bdirector\b", "finance ... director"),
    (r"\bextensive\b.*\blc-ms\b", "extensive ... lc-ms"),
    (r"\bextensive\b.*\bhplc\b", "extensive ... hplc"),
    (r"\bsales\b.*\brepresentative\b", "sales ... representative"),
    # Phrase + gap + word
    (r"\bgene\s+therapy\b.*\bnonclinical\b", "gene therapy ... nonclinical"),
    (r"\bdata\s+management\b.*\bpreclinical\b", "data management ... preclinical"),
    (r"\brare\s+disease\b.*\bdrug\s+development\b", "rare disease ... drug development"),
    (r"\bprotein\s+purification\b.*\bion\s+exchange\b", "protein purification ... ion exchange"),
    (r"\banalytical\s+chemistry\b.*\bmethod\s+development\b", "analytical chemistry ... method development"),
    (r"\bglp\b.*\btoxicology\b", "glp ... toxicology"),
    # Prefix match (no trailing \b)
    (r"\bbioinformatic", "bioinformatic*"),
    (r"\bcomputational\b.*\bbiolog", "computational ... biolog*"),
]


@pytest.mark.parametrize("regex,expected_display", ROUNDTRIP_CASES)
def test_regex_to_display(regex, expected_display):
    display, can_roundtrip = regex_to_display(regex)
    assert display == expected_display
    assert can_roundtrip is True


@pytest.mark.parametrize("regex,expected_display", ROUNDTRIP_CASES)
def test_display_to_regex(regex, expected_display):
    assert display_to_regex(expected_display) == regex


@pytest.mark.parametrize("regex,expected_display", ROUNDTRIP_CASES)
def test_roundtrip(regex, expected_display):
    """regex → display → regex must reproduce the original."""
    display, _ = regex_to_display(regex)
    assert display_to_regex(display) == regex


# ---------------------------------------------------------------------------
# Non-roundtrippable patterns (complex regex shown as-is)
# ---------------------------------------------------------------------------

NON_ROUNDTRIP_CASES = [
    r"\bpk\s*\bpd\b.*\bmodeling\b",       # \s* connector
    r"\blc-ms\s*\bms\b.*\bmethod\s+development\b",  # \s* connector
    r"\bbioreactor\b.*\bscale.*up\b",      # .* not between \b markers
    r"\bco-?op\b",                          # quantifier on hyphen
    r"\b(foo|bar)\b",                       # alternation group
]


@pytest.mark.parametrize("regex", NON_ROUNDTRIP_CASES)
def test_non_roundtrippable_marked(regex):
    _display, can_roundtrip = regex_to_display(regex)
    assert can_roundtrip is False


# ---------------------------------------------------------------------------
# display_to_regex: pass-through for raw regex input
# ---------------------------------------------------------------------------

def test_passthrough_raw_regex():
    """If display string already contains \\b or \\s, return as-is."""
    raw = r"\bfoo\s+bar\b"
    assert display_to_regex(raw) == raw


def test_passthrough_boundary_only():
    raw = r"\bfoo\b"
    assert display_to_regex(raw) == raw


def test_empty_string():
    assert display_to_regex("") == ""
    display, rt = regex_to_display("")
    assert display == ""
    assert rt is False


# ---------------------------------------------------------------------------
# Bulk test: every pattern in evaluator_patterns.yaml round-trips or is flagged
# ---------------------------------------------------------------------------

def test_all_yaml_patterns_handled():
    """Every pattern from the YAML file either round-trips or is flagged non-roundtrippable."""
    yaml_path = "data/evaluator_patterns.yaml"
    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        pytest.skip("evaluator_patterns.yaml not found")

    for key in ("skip_title_patterns", "skip_description_patterns",
                "rescue_patterns", "boost_patterns"):
        for pat in data.get(key, []):
            display, can_roundtrip = regex_to_display(pat)
            if can_roundtrip:
                # Must survive full round-trip
                assert display_to_regex(display) == pat, (
                    f"Round-trip failed for {pat!r}: display={display!r}, "
                    f"back={display_to_regex(display)!r}"
                )
            # Either way, the display should be a non-empty string
            assert display
            # The original regex must be valid
            re.compile(pat)
