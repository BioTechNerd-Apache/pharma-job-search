"""Convert between regex patterns and human-readable display strings.

The evaluator stores patterns as regex in YAML (e.g. ``\\bdrug\\s+substance\\b``).
This module provides two functions to convert between that format and a
plain-English display string (e.g. ``"drug substance"``), so the dashboard
can show patterns in a non-intimidating way while keeping storage unchanged.
"""

import re

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"\\b"          # word boundary
    r"|\\s\+"       # required whitespace
    r"|\\s\*"       # optional whitespace
    r"|\.\*"        # wildcard
    r"|[\w\-]+"     # literal word chars (including hyphens)
)


def _tokenize(regex_str: str) -> list[tuple[str, str]] | None:
    """Tokenize a regex string into (type, value) pairs.

    Returns None if the pattern contains unrecognized constructs.
    """
    tokens: list[tuple[str, str]] = []
    pos = 0
    s = regex_str

    while pos < len(s):
        m = _TOKEN_RE.match(s, pos)
        if not m:
            return None  # unrecognized construct
        val = m.group()
        if val == r"\b":
            tokens.append(("boundary", val))
        elif val == r"\s+":
            tokens.append(("space", val))
        elif val == r"\s*":
            tokens.append(("space_opt", val))
        elif val == ".*":
            tokens.append(("wildcard", val))
        else:
            tokens.append(("word", val))
        pos = m.end()

    return tokens


# ---------------------------------------------------------------------------
# Regex → Display
# ---------------------------------------------------------------------------

def regex_to_display(regex_str: str) -> tuple[str, bool]:
    """Convert a regex pattern to a human-readable display string.

    Returns ``(display_str, can_roundtrip)`` where *can_roundtrip* is True
    when ``display_to_regex(display_str) == regex_str`` is expected to hold.
    """
    tokens = _tokenize(regex_str)

    # Bail out if pattern has unrecognized constructs
    if tokens is None:
        return regex_str, False

    # Bail out if pattern uses \s* (non-standard spacing)
    if any(t[0] == "space_opt" for t in tokens):
        return regex_str, False

    # Must start with \b to be a standard pattern
    if not tokens or tokens[0] != ("boundary", r"\b"):
        return regex_str, False

    # Build display string from tokens
    parts: list[str] = []
    for ttype, _tval in tokens:
        if ttype == "boundary":
            pass  # skip word boundaries
        elif ttype == "word":
            parts.append(_tval)
        elif ttype == "space":
            parts.append(" ")
        elif ttype == "wildcard":
            parts.append(" ... ")

    display = "".join(parts).strip()

    # Detect prefix match: pattern ends with a word (no trailing \b)
    if tokens[-1][0] == "word":
        display += "*"

    # Clean up multiple spaces
    display = re.sub(r" {2,}", " ", display)

    # Verify actual round-trip
    can_roundtrip = display_to_regex(display) == regex_str
    return display, can_roundtrip


# ---------------------------------------------------------------------------
# Display → Regex
# ---------------------------------------------------------------------------

def display_to_regex(display_str: str) -> str:
    """Convert a human-readable display string back to a regex pattern.

    Rules:
    - If the string already contains ``\\b`` or ``\\s``, it is assumed to be
      raw regex and is returned as-is.
    - ``"word"`` → ``\\bword\\b``
    - ``"two words"`` → ``\\btwo\\s+words\\b``
    - ``"chief ... officer"`` → ``\\bchief\\b.*\\bofficer\\b``
    - ``"bioinformatic*"`` → ``\\bbioinformatic``  (prefix match, no trailing \\b)
    """
    s = display_str.strip()
    if not s:
        return s

    # Pass-through if user typed raw regex
    if r"\b" in s or r"\s" in s:
        return s

    # Check for trailing * (prefix-match on last word)
    prefix_match = s.endswith("*")
    if prefix_match:
        s = s[:-1].rstrip()

    # Split on " ... " (wildcard gap)
    segments = [seg.strip() for seg in re.split(r"\s*\.\.\.\s*", s) if seg.strip()]

    regex_parts: list[str] = []
    for i, seg in enumerate(segments):
        is_last = i == len(segments) - 1
        words = seg.split()
        if not words:
            continue

        # Build: \bw1\s+w2\s+...wN\b  (or no trailing \b if prefix on last segment)
        part = r"\b" + r"\s+".join(words)
        if prefix_match and is_last:
            pass  # no trailing \b
        else:
            part += r"\b"

        regex_parts.append(part)

    # Join segments with .*
    return ".*".join(regex_parts)
