"""Measure retrieval latency and labeled recall over the included synthetic job set."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from job_matcher import JobMatcher
from resume_rag import ResumeRAG


def run_benchmark(
    jobs_directory: str | Path = "data/jobs",
    ground_truth_path: str | Path = "data/evaluation_ground_truth.json",
    database_directory: str | Path = "vector_db",
    top_k: int = 10,
) -> dict[str, object]:
    jobs_root = Path(jobs_directory)
    ground_truth = json.loads(Path(ground_truth_path).read_text(encoding="utf-8"))
    matcher = JobMatcher(ResumeRAG(persist_directory=database_directory))
    per_job = []
    latencies = []
    total_expected = 0
    total_retrieved_relevant = 0

    for job_path in sorted(jobs_root.glob("*.txt")):
        description = job_path.read_text(encoding="utf-8")
        start = time.perf_counter()
        result = matcher.match(description, top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        expected = set(ground_truth.get(job_path.name, []))
        retrieved = {
            item["candidate_name"] for item in result["top_matches"]
        }
        hits = expected & retrieved
        total_expected += len(expected)
        total_retrieved_relevant += len(hits)
        per_job.append(
            {
                "job": job_path.name,
                "latency_ms": round(latency_ms, 2),
                "expected_relevant": len(expected),
                "retrieved_relevant": len(hits),
                "recall_at_k": round(len(hits) / len(expected), 3) if expected else None,
                "retrieved_candidates": len(retrieved),
            }
        )

    if not per_job:
        raise FileNotFoundError(f"No .txt job descriptions found under: {jobs_root}")
    return {
        "top_k": top_k,
        "job_count": len(per_job),
        "macro_recall_at_k": round(
            statistics.mean(job["recall_at_k"] for job in per_job if job["recall_at_k"] is not None),
            3,
        ),
        "micro_recall_at_k": round(total_retrieved_relevant / total_expected, 3) if total_expected else None,
        "mean_latency_ms": round(statistics.mean(latencies), 2),
        "p95_latency_ms": round(sorted(latencies)[min(len(latencies) - 1, int(0.95 * len(latencies)))], 2),
        "per_job": per_job,
        "note": "Recall is measured against the curated synthetic relevance labels, not human-reviewed production data.",
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2))
