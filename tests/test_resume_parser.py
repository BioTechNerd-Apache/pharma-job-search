"""Tests for resume text extraction."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.resume_parser import extract_text, _clean_text, SUPPORTED_EXTENSIONS, MAX_CHARS


class TestExtractText:
    """Tests for the main extract_text function."""

    def test_txt_extraction(self, tmp_path):
        """Extract text from a plain text file."""
        resume = tmp_path / "resume.txt"
        resume.write_text("Jane Smith, PhD\nSenior Research Scientist\n10 years experience in bioanalytical chemistry")
        text = extract_text(str(resume))
        assert "Jane Smith" in text
        assert "bioanalytical" in text

    def test_txt_extraction_unicode(self, tmp_path):
        """Handle unicode characters in text files."""
        resume = tmp_path / "resume.txt"
        resume.write_text("Dr. M\u00fcller - Research Scientist\nExperience with qPCR and flow cytometry at Universit\u00e4t Z\u00fcrich")
        text = extract_text(str(resume))
        assert "M\u00fcller" in text

    def test_file_not_found(self):
        """Raise ValueError for non-existent file."""
        with pytest.raises(ValueError, match="File not found"):
            extract_text("/nonexistent/path/resume.pdf")

    def test_unsupported_extension(self, tmp_path):
        """Raise ValueError for unsupported file types."""
        bad_file = tmp_path / "resume.jpg"
        bad_file.write_text("not a resume")
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text(str(bad_file))

    def test_empty_file(self, tmp_path):
        """Raise ValueError for empty files."""
        resume = tmp_path / "empty.txt"
        resume.write_text("")
        with pytest.raises(ValueError, match="Could not extract meaningful text"):
            extract_text(str(resume))

    def test_too_short_content(self, tmp_path):
        """Raise ValueError for files with too little content."""
        resume = tmp_path / "short.txt"
        resume.write_text("Hi")
        with pytest.raises(ValueError, match="Could not extract meaningful text"):
            extract_text(str(resume))

    def test_truncation(self, tmp_path):
        """Truncate text longer than MAX_CHARS."""
        resume = tmp_path / "long.txt"
        resume.write_text("A" * (MAX_CHARS + 5000))
        text = extract_text(str(resume))
        assert len(text) <= MAX_CHARS

    def test_supported_extensions(self):
        """Verify supported extensions set."""
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".jpg" not in SUPPORTED_EXTENSIONS


class TestCleanText:
    """Tests for text cleaning."""

    def test_collapse_blank_lines(self):
        result = _clean_text("Line 1\n\n\n\n\nLine 2")
        assert result == "Line 1\n\nLine 2"

    def test_collapse_spaces(self):
        result = _clean_text("Too   many    spaces")
        assert result == "Too many spaces"

    def test_strip_lines(self):
        result = _clean_text("  leading space  \n  trailing space  ")
        assert result == "leading space\ntrailing space"

    def test_strip_overall(self):
        result = _clean_text("\n\n  Hello World  \n\n")
        assert result == "Hello World"
