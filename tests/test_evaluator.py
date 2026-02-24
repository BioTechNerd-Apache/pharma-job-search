"""Tests for the job evaluation pre-filter (Stage 1)."""

import sys
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluator import prefilter_job, reload_patterns


@pytest.fixture(autouse=True)
def use_builtin_patterns():
    """Force builtin patterns so tests aren't affected by local YAML overrides."""
    reload_patterns("/nonexistent/path.yaml")
    yield
    reload_patterns()


class TestSkipTitles:
    """Titles that should be auto-skipped."""

    def test_skip_vp(self):
        result = prefilter_job("VP of Research Operations")
        assert result.action == "skip"

    def test_skip_svp(self):
        result = prefilter_job("SVP Bioanalytical Sciences")
        assert result.action == "skip"

    def test_skip_hplc_scientist(self):
        result = prefilter_job("HPLC Scientist - Analytical Development")
        assert result.action == "skip"

    def test_skip_lcms(self):
        result = prefilter_job("LC-MS Bioanalytical Scientist")
        assert result.action == "skip"

    def test_skip_qc_analyst(self):
        result = prefilter_job("QC Analyst - Release Testing")
        assert result.action == "skip"

    def test_skip_research_associate(self):
        result = prefilter_job("Research Associate - Cell Biology")
        assert result.action == "skip"

    def test_skip_lab_technician(self):
        result = prefilter_job("Lab Technician - Molecular Biology")
        assert result.action == "skip"

    def test_skip_bioinformatics(self):
        result = prefilter_job("Bioinformatics Scientist")
        assert result.action == "skip"

    def test_skip_computational_biologist(self):
        result = prefilter_job("Computational Biologist")
        assert result.action == "skip"

    def test_skip_data_scientist(self):
        result = prefilter_job("Data Scientist - Genomics")
        assert result.action == "skip"

    def test_skip_organic_chemist(self):
        result = prefilter_job("Organic Chemist - Process Development")
        assert result.action == "skip"

    def test_skip_medicinal_chemist(self):
        result = prefilter_job("Medicinal Chemist")
        assert result.action == "skip"

    def test_skip_postdoc(self):
        result = prefilter_job("Postdoctoral Fellow - Cancer Biology")
        assert result.action == "skip"

    def test_skip_intern(self):
        result = prefilter_job("Intern - Molecular Biology Summer 2026")
        assert result.action == "skip"

    def test_skip_sales(self):
        result = prefilter_job("Sales Representative - Pharma")
        assert result.action == "skip"

    def test_skip_manufacturing_technician(self):
        result = prefilter_job("Manufacturing Technician - Cell Therapy")
        assert result.action == "skip"

    def test_skip_protein_scientist(self):
        result = prefilter_job("Protein Scientist - Biologics")
        assert result.action == "skip"

    def test_skip_formulation_scientist(self):
        result = prefilter_job("Formulation Scientist - Drug Product")
        assert result.action == "skip"

    def test_skip_msl(self):
        result = prefilter_job("Medical Science Liaison - Oncology")
        assert result.action == "skip"

    def test_skip_project_manager(self):
        result = prefilter_job("Project Manager - Clinical Operations")
        assert result.action == "skip"

    def test_skip_analytical_chemist(self):
        result = prefilter_job("Analytical Chemist - QC")
        assert result.action == "skip"


class TestPassTitles:
    """Titles that should NOT be skipped."""

    def test_pass_senior_scientist(self):
        result = prefilter_job("Senior Scientist - Bioanalytical")
        assert result.action != "skip"

    def test_pass_principal_scientist(self):
        result = prefilter_job("Principal Scientist - Gene Therapy")
        assert result.action != "skip"

    def test_pass_associate_director(self):
        result = prefilter_job("Associate Director - Bioanalytical Sciences")
        assert result.action != "skip"

    def test_pass_director_molecular_biology(self):
        result = prefilter_job("Director - Molecular Biology")
        assert result.action != "skip"

    def test_pass_scientist_flow_cytometry(self):
        result = prefilter_job("Scientist - Flow Cytometry")
        assert result.action != "skip"

    def test_pass_method_validation(self):
        result = prefilter_job("Scientist - Method Validation")
        assert result.action != "skip"

    def test_pass_cell_therapy_scientist(self):
        result = prefilter_job("Scientist - Cell Therapy Analytics")
        assert result.action != "skip"

    def test_pass_study_director(self):
        result = prefilter_job("Study Director - Bioanalytical")
        assert result.action != "skip"

    def test_pass_staff_scientist(self):
        result = prefilter_job("Staff Scientist - Oncology Research")
        assert result.action != "skip"


class TestBoostPatterns:
    """Jobs that should be boosted (evaluated with priority)."""

    def test_boost_bioanalytical_title(self):
        result = prefilter_job("Senior Scientist - Bioanalytical")
        assert result.action == "boost"

    def test_boost_gene_therapy(self):
        result = prefilter_job("Scientist - Gene Therapy Bioanalytical")
        assert result.action == "boost"

    def test_boost_car_t(self):
        result = prefilter_job("Scientist - CAR-T Analytics")
        assert result.action == "boost"

    def test_boost_flow_cytometry(self):
        result = prefilter_job("Scientist - Flow Cytometry")
        assert result.action == "boost"

    def test_boost_method_validation(self):
        result = prefilter_job("Scientist - Method Validation")
        assert result.action == "boost"

    def test_boost_qpcr_in_description(self):
        result = prefilter_job("Senior Scientist", "Experience with qPCR and ddPCR methods")
        assert result.action == "boost"

    def test_boost_viral_vector(self):
        result = prefilter_job("Scientist", "Work with AAV viral vector characterization")
        assert result.action == "boost"

    def test_boost_glp(self):
        result = prefilter_job("Scientist", "GLP method validation for biodistribution studies")
        assert result.action == "boost"


class TestRescueLogic:
    """Jobs that match skip description patterns but are rescued."""

    def test_rescue_hplc_but_qpcr(self):
        """Job mentions HPLC extensively but also requires qPCR — should not skip."""
        result = prefilter_job(
            "Senior Scientist - Analytical",
            "Must have extensive experience with HPLC and qPCR methods for viral vector characterization"
        )
        assert result.action != "skip"

    def test_rescue_cho_but_gene_therapy(self):
        """Description mentions CHO cells but also gene therapy — should not skip."""
        result = prefilter_job(
            "Scientist - Analytical Development",
            "CHO cell line development for gene therapy products. Experience with ddPCR."
        )
        assert result.action != "skip"

    def test_rescue_ada_but_flow_cytometry(self):
        """Description mentions ADA assay but also flow cytometry — should not skip."""
        result = prefilter_job(
            "Scientist",
            "ADA assay development and flow cytometry for cell therapy monitoring"
        )
        assert result.action != "skip"

    def test_no_rescue_pure_hplc(self):
        """Pure HPLC role with no rescue keywords — should skip."""
        result = prefilter_job(
            "Senior Scientist",
            "Must have extensive experience with HPLC for protein characterization. SEC-HPLC, IEX."
        )
        assert result.action == "skip"

    def test_no_rescue_pure_bioreactor(self):
        """Pure bioreactor role — should skip."""
        result = prefilter_job(
            "Scientist",
            "Bioreactor operation and cell culture scale-up for manufacturing."
        )
        assert result.action == "skip"


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_title(self):
        result = prefilter_job("")
        assert result.action == "evaluate"

    def test_none_title(self):
        result = prefilter_job(None)
        assert result.action == "evaluate"

    def test_empty_description(self):
        result = prefilter_job("Scientist", "")
        assert result.action == "evaluate"

    def test_case_insensitive_skip(self):
        result = prefilter_job("BIOINFORMATICS SCIENTIST")
        assert result.action == "skip"

    def test_case_insensitive_boost(self):
        result = prefilter_job("scientist - BIOANALYTICAL DEVELOPMENT")
        assert result.action == "boost"
