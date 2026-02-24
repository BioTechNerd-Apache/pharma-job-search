"""Tests for the setup wizard logic."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.setup_wizard import WizardOutput, save_wizard_output, _backup_file, _check_api_key_status


class TestWizardOutput:
    """Tests for WizardOutput dataclass."""

    def test_default_values(self):
        output = WizardOutput()
        assert output.resume_profile == {}
        assert output.search_terms == []
        assert output.evaluator_patterns == {}
        assert output.api_keys == {}

    def test_populated_values(self):
        output = WizardOutput(
            resume_profile={"name": "Test User"},
            search_terms=["scientist", "bioanalytical"],
            priority_terms=["bioanalytical"],
            synonyms={"bioanalytical": ["bioanalysis"]},
            filter_include=["biolog"],
            filter_exclude=["software"],
            evaluator_patterns={"skip_title_patterns": [r"\bvp\b"]},
            api_keys={"anthropic_api_key": "sk-test"},
        )
        assert output.resume_profile["name"] == "Test User"
        assert len(output.search_terms) == 2
        assert "bioanalytical" in output.priority_terms


class TestBackupFile:
    """Tests for file backup logic."""

    def test_backup_existing_file(self, tmp_path):
        original = tmp_path / "test.json"
        original.write_text('{"key": "value"}')
        _backup_file(original)
        bak = tmp_path / "test.json.bak"
        assert bak.exists()
        assert bak.read_text() == '{"key": "value"}'

    def test_backup_nonexistent_file(self, tmp_path):
        """No error when backing up a file that doesn't exist."""
        _backup_file(tmp_path / "nonexistent.json")
        # Should not raise


class TestSaveWizardOutput:
    """Tests for saving wizard output files."""

    def test_save_resume_profile(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.setup_wizard.PROJECT_ROOT", tmp_path)
        monkeypatch.setattr("src.setup_wizard.DEFAULT_CONFIG_PATH", tmp_path / "config.yaml")
        (tmp_path / "data").mkdir()

        output = WizardOutput(
            resume_profile={"name": "Test User", "years_experience": 5},
        )
        files = save_wizard_output(output, {})
        assert any("resume_profile.json" in f for f in files)

        profile_path = tmp_path / "data" / "resume_profile.json"
        assert profile_path.exists()
        data = json.loads(profile_path.read_text())
        assert data["name"] == "Test User"

    def test_save_evaluator_patterns(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.setup_wizard.PROJECT_ROOT", tmp_path)
        monkeypatch.setattr("src.setup_wizard.DEFAULT_CONFIG_PATH", tmp_path / "config.yaml")
        (tmp_path / "data").mkdir()

        output = WizardOutput(
            resume_profile={"name": "Test"},
            evaluator_patterns={
                "skip_title_patterns": [r"\bvp\b", r"\bsales\b"],
                "skip_description_patterns": [],
                "rescue_patterns": [r"\bqpcr\b"],
                "boost_patterns": [r"\bbioanalytical\b"],
            },
        )
        files = save_wizard_output(output, {})
        assert any("evaluator_patterns.yaml" in f for f in files)

        patterns_path = tmp_path / "data" / "evaluator_patterns.yaml"
        assert patterns_path.exists()
        data = yaml.safe_load(patterns_path.read_text())
        assert r"\bvp\b" in data["skip_title_patterns"]

    def test_save_config_yaml(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.setup_wizard.PROJECT_ROOT", tmp_path)
        monkeypatch.setattr("src.setup_wizard.DEFAULT_CONFIG_PATH", tmp_path / "config.yaml")
        (tmp_path / "data").mkdir()

        output = WizardOutput(
            resume_profile={"name": "Test"},
            search_terms=["scientist", "bioanalytical scientist"],
            filter_include=["biolog", "pharma"],
            filter_exclude=["software"],
            api_keys={"anthropic_api_key": "sk-test123"},
        )
        files = save_wizard_output(output, {})
        assert any("config.yaml" in f for f in files)

        config_path = tmp_path / "config.yaml"
        data = yaml.safe_load(config_path.read_text())
        assert "scientist" in data["search"]["terms"]
        assert "biolog" in data["search"]["filter_include"]
        assert data["evaluation"]["anthropic_api_key"] == "sk-test123"

    def test_preserves_existing_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.setup_wizard.PROJECT_ROOT", tmp_path)
        monkeypatch.setattr("src.setup_wizard.DEFAULT_CONFIG_PATH", tmp_path / "config.yaml")
        (tmp_path / "data").mkdir()

        existing = {
            "search": {"days": 14, "sites": ["indeed"]},
            "usajobs": {"api_key": "existing-key"},
        }
        output = WizardOutput(
            resume_profile={"name": "Test"},
            search_terms=["new term"],
        )
        save_wizard_output(output, existing)

        config_path = tmp_path / "config.yaml"
        data = yaml.safe_load(config_path.read_text())
        # Existing keys preserved
        assert data["usajobs"]["api_key"] == "existing-key"
        # New terms applied
        assert "new term" in data["search"]["terms"]


class TestCheckApiKeyStatus:
    """Tests for API key status checking."""

    def test_empty_config(self):
        status = _check_api_key_status({})
        assert status["anthropic"] == ""
        assert status["usajobs_key"] == ""
        assert status["jooble"] == ""

    def test_configured_keys(self):
        config = {
            "evaluation": {"anthropic_api_key": "sk-test"},
            "usajobs": {"api_key": "usa-key", "email": "test@example.com"},
            "adzuna": {"app_id": "adz-id", "app_key": "adz-key"},
            "jooble": {"api_key": "joo-key"},
        }
        status = _check_api_key_status(config)
        assert status["anthropic"] == "sk-test"
        assert status["usajobs_key"] == "usa-key"
        assert status["adzuna_app_id"] == "adz-id"
        assert status["jooble"] == "joo-key"

    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
        status = _check_api_key_status({})
        assert status["anthropic"] == "sk-env"


class TestEvaluatorPatternsLoading:
    """Tests for evaluator pattern loading with YAML."""

    def test_builtin_fallback(self):
        """Without YAML file, built-in patterns are used."""
        from src.evaluator import load_evaluator_patterns
        patterns = load_evaluator_patterns("/nonexistent/path.yaml")
        assert len(patterns["skip_title_patterns"]) > 0
        assert len(patterns["boost_patterns"]) > 0

    def test_yaml_loading(self, tmp_path):
        """YAML file patterns are loaded correctly."""
        from src.evaluator import load_evaluator_patterns
        yaml_path = tmp_path / "patterns.yaml"
        data = {
            "skip_title_patterns": [r"\bcustom\b"],
            "skip_description_patterns": [r"\btest\b"],
            "rescue_patterns": [r"\brescue\b"],
            "boost_patterns": [r"\bboost\b"],
        }
        yaml_path.write_text(yaml.dump(data))

        patterns = load_evaluator_patterns(str(yaml_path))
        assert patterns["skip_title_patterns"] == [r"\bcustom\b"]
        assert patterns["boost_patterns"] == [r"\bboost\b"]

    def test_invalid_regex_skipped(self, tmp_path):
        """Invalid regex patterns in YAML are skipped."""
        from src.evaluator import load_evaluator_patterns
        yaml_path = tmp_path / "patterns.yaml"
        data = {
            "skip_title_patterns": [r"\bvalid\b", "[invalid", r"\balso_valid\b"],
            "skip_description_patterns": [],
            "rescue_patterns": [],
            "boost_patterns": [],
        }
        yaml_path.write_text(yaml.dump(data))

        patterns = load_evaluator_patterns(str(yaml_path))
        assert r"\bvalid\b" in patterns["skip_title_patterns"]
        assert r"\balso_valid\b" in patterns["skip_title_patterns"]
        assert "[invalid" not in patterns["skip_title_patterns"]

    def test_partial_yaml_uses_defaults(self, tmp_path):
        """YAML with only some keys uses defaults for missing keys."""
        from src.evaluator import load_evaluator_patterns, _BUILTIN_BOOST_PATTERNS
        yaml_path = tmp_path / "patterns.yaml"
        data = {
            "skip_title_patterns": [r"\bcustom\b"],
        }
        yaml_path.write_text(yaml.dump(data))

        patterns = load_evaluator_patterns(str(yaml_path))
        assert patterns["skip_title_patterns"] == [r"\bcustom\b"]
        # Missing keys fall back to built-in
        assert patterns["boost_patterns"] == list(_BUILTIN_BOOST_PATTERNS)
