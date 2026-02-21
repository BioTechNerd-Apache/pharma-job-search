"""Persistent storage for job evaluation results (evaluations.json)."""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import EvaluationConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)


class EvaluationStore:
    """Manages evaluations.json — keyed by job_url to prevent re-evaluation."""

    def __init__(self, config: EvaluationConfig):
        self.path = PROJECT_ROOT / config.evaluations_store
        self.model = config.model
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load evaluations: {e}")
        return {"metadata": {}, "evaluations": {}}

    def save(self):
        """Persist evaluations to disk."""
        self._data["metadata"]["last_evaluation_run"] = datetime.now().isoformat()
        self._data["metadata"]["total_evaluated"] = len(self._data["evaluations"])
        self._data["metadata"]["model_used"] = self.model
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)
        logger.info(f"Evaluations saved: {self.path} ({len(self._data['evaluations'])} entries)")

    def is_evaluated(self, job_url: str) -> bool:
        return job_url in self._data.get("evaluations", {})

    def add_evaluation(self, job_url: str, evaluation: dict):
        if "evaluations" not in self._data:
            self._data["evaluations"] = {}
        self._data["evaluations"][job_url] = evaluation

    def get_evaluation(self, job_url: str) -> dict | None:
        return self._data.get("evaluations", {}).get(job_url)

    def get_all_evaluations(self) -> dict:
        return self._data.get("evaluations", {})

    def get_unevaluated(self, job_urls: list[str]) -> list[str]:
        """Return job_urls that haven't been evaluated yet."""
        evaluated = self._data.get("evaluations", {})
        return [url for url in job_urls if url not in evaluated]

    def merge_to_dataframe(self, jobs_df: pd.DataFrame) -> pd.DataFrame:
        """Merge evaluation results into a jobs DataFrame.

        Joins on job_url, adding evaluation columns to the DataFrame.
        Only returns rows that have evaluations.
        """
        evals = self._data.get("evaluations", {})
        if not evals:
            return pd.DataFrame()

        eval_records = []
        for url, ev in evals.items():
            record = {"job_url": url}
            record.update(ev)
            eval_records.append(record)

        eval_df = pd.DataFrame(eval_records)

        if jobs_df.empty or eval_df.empty:
            return eval_df

        # Merge with job data to get title, company, location, etc.
        merged = jobs_df.merge(eval_df, on="job_url", how="inner")
        return merged

    def export_csv(self, jobs_df: pd.DataFrame, output_path: str, min_score: int = 0) -> Path:
        """Export evaluation results to CSV, optionally filtered by min score."""
        merged = self.merge_to_dataframe(jobs_df)
        if merged.empty:
            logger.warning("No evaluation data to export")
            return Path(output_path)

        if min_score > 0:
            merged = merged[merged["fit_score"] >= min_score]

        merged = merged.sort_values("fit_score", ascending=False)
        path = Path(output_path)
        merged.to_csv(path, index=False)
        logger.info(f"Exported {len(merged)} evaluations to {path}")
        return path

    def summary(self) -> dict:
        """Return summary statistics of evaluations."""
        evals = self._data.get("evaluations", {})
        if not evals:
            return {"total": 0}

        scores = []
        buckets = {"strong": 0, "moderate": 0, "weak": 0, "poor": 0,
                    "prefilter_skip": 0, "error": 0}
        recommendations = {"apply": 0, "maybe": 0, "skip": 0}
        total_input_tokens = 0
        total_output_tokens = 0

        for ev in evals.values():
            score = ev.get("fit_score", 0)
            bucket = ev.get("fit_bucket", "poor")
            rec = ev.get("recommendation", "skip")
            scores.append(score)
            buckets[bucket] = buckets.get(bucket, 0) + 1
            recommendations[rec] = recommendations.get(rec, 0) + 1
            total_input_tokens += ev.get("input_tokens", 0)
            total_output_tokens += ev.get("output_tokens", 0)

        api_evaluated = [s for s, ev in zip(scores, evals.values())
                         if ev.get("fit_bucket") != "prefilter_skip"]

        return {
            "total": len(evals),
            "buckets": buckets,
            "recommendations": recommendations,
            "avg_score": round(sum(api_evaluated) / len(api_evaluated), 1) if api_evaluated else 0,
            "max_score": max(scores) if scores else 0,
            "min_score_api": min(api_evaluated) if api_evaluated else 0,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "metadata": self._data.get("metadata", {}),
        }
