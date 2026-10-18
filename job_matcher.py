"""Hybrid semantic and skill-based job-to-resume matching."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from resume_rag import ResumeRAG, extract_skills


def parse_requirements(job_description: str) -> dict[str, Any]:
    """Identify explicit must-have skills and minimum years from a job description."""
    explicit_skills: list[str] = []
    for line in job_description.splitlines():
        if re.search(r"\b(?:must[- ]have|required)\b", line, re.IGNORECASE):
            explicit_skills.extend(extract_skills(line))

    experience_by_skill: dict[str, float] = {}
    for match in re.finditer(
        r"\b(\d+(?:\.\d+)?)\s*\+?\s+years?\s+(?:of\s+)?(?:experience\s+)?(?:with\s+)?([A-Za-z][A-Za-z0-9+#./ -]{0,35})",
        job_description,
        re.IGNORECASE,
    ):
        years = float(match.group(1))
        trailing = match.group(2)
        skills = extract_skills(trailing)
        for skill in skills:
            experience_by_skill[skill] = max(experience_by_skill.get(skill, 0.0), years)

    generic_years = [
        float(value)
        for value in re.findall(
            r"\b(\d+(?:\.\d+)?)\s*\+?\s+years?\s+(?:of\s+)?(?:relevant\s+)?experience\b",
            job_description,
            re.IGNORECASE,
        )
    ]
    return {
        "must_have_skills": list(dict.fromkeys(explicit_skills + list(experience_by_skill))),
        "minimum_experience_years": max(generic_years, default=0.0),
        "experience_by_skill": experience_by_skill,
        "mentioned_skills": extract_skills(job_description),
    }


class JobMatcher:
    def __init__(self, rag: ResumeRAG) -> None:
        self.rag = rag

    def match(
        self,
        job_description: str,
        top_k: int = 10,
        must_have_skills: list[str] | None = None,
        minimum_experience_years: float | None = None,
    ) -> dict[str, Any]:
        if not job_description.strip():
            raise ValueError("Job description cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if self.rag.collection.count() == 0:
            raise ValueError("No resumes are indexed. Run resume_rag.py ingest first.")

        requirements = parse_requirements(job_description)
        if must_have_skills is not None:
            requirements["must_have_skills"] = list(dict.fromkeys(must_have_skills))
        if minimum_experience_years is not None:
            if minimum_experience_years < 0:
                raise ValueError("minimum_experience_years cannot be negative.")
            requirements["minimum_experience_years"] = minimum_experience_years

        query = self.rag.embed_query(job_description)
        result = self.rag.collection.query(
            query_embeddings=[query],
            n_results=self.rag.collection.count(),
            include=["documents", "metadatas", "distances"],
        )

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for document, metadata, distance in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            grouped[metadata["source_key"]].append(
                {"document": document, "metadata": metadata, "distance": distance}
            )

        matches = []
        required_skills = requirements["must_have_skills"]
        for chunks in grouped.values():
            metadata = chunks[0]["metadata"]
            candidate_skills = [
                skill for skill in (metadata.get("skills") or "").split(", ") if skill
            ]
            skill_set = {skill.casefold() for skill in candidate_skills}
            missing = [skill for skill in required_skills if skill.casefold() not in skill_set]
            candidate_years = float(metadata.get("experience_years") or 0)
            if missing or candidate_years < requirements["minimum_experience_years"]:
                continue
            if any(candidate_years < years for years in requirements["experience_by_skill"].values()):
                continue

            best_chunks = sorted(chunks, key=lambda item: item["distance"])
            semantic_score = max(0.0, min(1.0, 1.0 - float(best_chunks[0]["distance"])))
            mentioned = requirements["mentioned_skills"]
            matched_skills = [
                skill for skill in candidate_skills
                if skill.casefold() in {item.casefold() for item in mentioned}
            ]
            skill_score = len(matched_skills) / max(1, len(mentioned))
            query_terms = {
                token.casefold()
                for token in re.findall(r"\b[A-Za-z][A-Za-z0-9+#.-]{2,}\b", job_description)
            }
            candidate_terms = {
                token.casefold()
                for chunk in chunks
                for token in re.findall(r"\b[A-Za-z][A-Za-z0-9+#.-]{2,}\b", chunk["document"])
            }
            lexical_score = len(query_terms & candidate_terms) / max(1, len(query_terms))
            score = round(100 * (0.55 * semantic_score + 0.35 * skill_score + 0.10 * lexical_score))
            relevant = []
            for item in best_chunks:
                excerpt = item["document"]
                if excerpt not in relevant:
                    relevant.append(excerpt)
                if len(relevant) == 3:
                    break
            reasoning_parts = [
                f"Semantic similarity {semantic_score:.0%}",
                f"{len(matched_skills)} of {len(mentioned)} mentioned skills matched",
                f"{candidate_years:g} years of experience",
                f"Relevant sections: {', '.join(dict.fromkeys(item['metadata']['section'] for item in best_chunks[:3]))}",
            ]
            matches.append(
                {
                    "candidate_name": metadata["candidate_name"],
                    "resume_path": metadata["resume_path"],
                    "match_score": max(0, min(100, score)),
                    "matched_skills": matched_skills,
                    "relevant_excerpts": relevant,
                    "reasoning": "; ".join(reasoning_parts) + ".",
                    "_semantic_score": semantic_score,
                    "_skill_score": skill_score,
                }
            )

        matches.sort(key=lambda item: item["match_score"], reverse=True)
        top_matches = []
        for candidate in matches[:top_k]:
            top_matches.append({key: value for key, value in candidate.items() if not key.startswith("_")})
        return {"job_description": job_description, "top_matches": top_matches}


def main() -> None:
    parser = argparse.ArgumentParser(description="Match resumes to a job description.")
    parser.add_argument("--job", required=True, help="Path to a job-description text file.")
    parser.add_argument("--db", default="vector_db", help="ChromaDB persistence directory.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--must-have", action="append", default=None, help="Required skill; repeat as needed.")
    parser.add_argument("--min-years", type=float, default=None)
    args = parser.parse_args()

    description = Path(args.job).read_text(encoding="utf-8")
    result = JobMatcher(ResumeRAG(persist_directory=args.db)).match(
        description,
        top_k=args.top_k,
        must_have_skills=args.must_have,
        minimum_experience_years=args.min_years,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
