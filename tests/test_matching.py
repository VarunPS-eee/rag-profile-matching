from pathlib import Path

import pytest

from job_matcher import JobMatcher, parse_requirements
from resume_rag import chunk_sections, extract_metadata, extract_skills, parse_sections


def test_chunking_keeps_sections_separate_and_labels_chunks():
    chunks = chunk_sections(
        [("Experience", "one two three four five"), ("Education", "six seven")],
        chunk_size=3,
        overlap=1,
    )

    assert [chunk["section"] for chunk in chunks] == ["Experience", "Experience", "Education"]
    assert all(chunk["text"].startswith(f"{chunk['section']}: ") for chunk in chunks)
    assert chunks[0]["text"] == "Experience: one two three"
    assert chunks[1]["text"] == "Experience: three four five"


def test_chunking_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_sections([("Skills", "Python")], chunk_size=10, overlap=10)


def test_section_parser_recognizes_common_headings():
    sections = parse_sections("Jane Doe\n\nExperience:\nBuilt APIs.\n\nEducation\nB.S. Computer Science")

    assert sections == [
        ("General", "Jane Doe"),
        ("Experience", "Built APIs."),
        ("Education", "B.S. Computer Science"),
    ]


def test_metadata_extracts_name_skills_years_and_education(tmp_path: Path):
    content = """Maya Chen
Skills
Python, RAG, NLP
Experience
7 years of experience delivering NLP products
Education
M.S. Computer Science"""

    metadata = extract_metadata(content, tmp_path / "resume.txt")

    assert metadata["candidate_name"] == "Maya Chen"
    assert metadata["experience_years"] == 7
    assert set(metadata["skills"].split(", ")) == {"Python", "RAG", "NLP"}
    assert metadata["education"] == "M.S. Computer Science"


def test_skill_extraction_deduplicates_nlp_alias():
    assert extract_skills("Natural Language Processing, NLP, Python") == ["NLP", "Python"]


def test_requirements_detect_explicit_skills_and_minimum_years():
    requirements = parse_requirements(
        "Minimum 5+ years of experience.\nMust-have skills: Python, SQL"
    )

    assert requirements["must_have_skills"] == ["Python", "SQL"]
    assert requirements["minimum_experience_years"] == 5


def test_skill_specific_year_requirement_becomes_a_must_have():
    requirements = parse_requirements("We need 5+ years Python experience.")

    assert requirements["must_have_skills"] == ["Python"]
    assert requirements["experience_by_skill"] == {"Python": 5}


class FakeCollection:
    def count(self):
        return 2

    def query(self, **kwargs):
        return {
            "documents": [[
                "Experience: Built Python machine learning models.",
                "Skills: Java, Spring, PostgreSQL",
            ]],
            "metadatas": [[
                {
                    "source_key": "maya",
                    "candidate_name": "Maya Chen",
                    "resume_path": "resumes/maya.txt",
                    "skills": "Python, Machine Learning",
                    "experience_years": 7,
                    "section": "Experience",
                },
                {
                    "source_key": "liam",
                    "candidate_name": "Liam O'Connor",
                    "resume_path": "resumes/liam.txt",
                    "skills": "Java, Spring, PostgreSQL",
                    "experience_years": 9,
                    "section": "Skills",
                },
            ]],
            "distances": [[0.1, 0.8]],
        }


class FakeRAG:
    collection = FakeCollection()

    @staticmethod
    def embed_query(text):
        return [1.0]


def test_matcher_filters_required_skills_and_returns_output_schema():
    result = JobMatcher(FakeRAG()).match(
        "Must-have skills: Python\nLooking for a machine learning engineer.",
        top_k=10,
    )

    assert result["job_description"].startswith("Must-have")
    assert len(result["top_matches"]) == 1
    assert result["top_matches"][0]["candidate_name"] == "Maya Chen"
    assert result["top_matches"][0]["matched_skills"] == ["Python", "Machine Learning"]
    assert 0 <= result["top_matches"][0]["match_score"] <= 100
    assert result["top_matches"][0]["relevant_excerpts"]


def test_matcher_enforces_global_experience_requirement():
    result = JobMatcher(FakeRAG()).match(
        "Must-have skills: Python\nMinimum 8+ years of experience."
    )

    assert result["top_matches"] == []
