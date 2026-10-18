"""Resume ingestion, section-aware chunking, metadata extraction, and ChromaDB storage."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
SECTION_NAMES = {
    "summary": "Summary",
    "profile": "Summary",
    "professional summary": "Summary",
    "objective": "Summary",
    "experience": "Experience",
    "work experience": "Experience",
    "professional experience": "Experience",
    "employment history": "Experience",
    "education": "Education",
    "academic background": "Education",
    "skills": "Skills",
    "technical skills": "Skills",
    "core competencies": "Skills",
    "projects": "Projects",
    "certifications": "Certifications",
}
SKILL_NAMES = (
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust", "SQL",
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "AWS", "Azure", "GCP", "Docker",
    "Kubernetes", "Terraform", "Linux", "React", "Angular", "Node.js", "Django",
    "FastAPI", "Flask", "Spring", "Machine Learning", "Deep Learning", "NLP",
    "Natural Language Processing", "LLM", "Generative AI", "PyTorch", "TensorFlow",
    "scikit-learn", "Pandas", "NumPy", "Spark", "Hadoop", "Airflow", " dbt ",
    "Tableau", "Power BI", "Excel", "Data Analysis", "Data Engineering", "ETL",
    "REST API", "GraphQL", "Microservices", "Git", "CI/CD", "Jenkins", "GitHub Actions",
    "Selenium", "Playwright", "Jest", "Agile", "Scrum", "Product Management",
    "Project Management", "Figma", "UX Research", "Salesforce", "Communication",
    "Leadership", "Statistics", "MLOps", "RAG", "Vector Databases", "ChromaDB",
    "OpenAI", "Hugging Face", "LangChain", "CSS", "HTML", "Kotlin", "Android",
    "Prometheus", "Kafka", "BigQuery", "Security",
)


def _skill_pattern(skill: str) -> re.Pattern[str]:
    normalized = skill.strip()
    escaped = re.escape(normalized)
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)


def extract_skills(text: str) -> list[str]:
    """Return recognized skills in a stable canonical form."""
    aliases = {"Natural Language Processing": "NLP"}
    found = []
    for skill in SKILL_NAMES:
        match = _skill_pattern(skill).search(text)
        if match:
            canonical = aliases.get(skill.strip(), skill.strip())
            found.append((match.start(), canonical))
    ordered = []
    for _, skill in sorted(found):
        if skill not in ordered:
            ordered.append(skill)
    return ordered


def parse_sections(text: str) -> list[tuple[str, str]]:
    """Split a resume on recognizable headings without mixing section boundaries."""
    sections: list[tuple[str, list[str]]] = []
    current = "General"
    sections.append((current, []))

    for line in text.splitlines():
        heading = re.sub(r"[\s:|•\-]+$", "", line.strip()).lower()
        heading = re.sub(r"^\W+|\W+$", "", heading)
        normalized = SECTION_NAMES.get(heading)
        if normalized:
            current = normalized
            if not sections or sections[-1][0] != current:
                sections.append((current, []))
            continue
        sections[-1][1].append(line.strip())

    result = []
    for name, lines in sections:
        content = "\n".join(line for line in lines if line).strip()
        if content:
            result.append((name, content))
    return result or [("General", text.strip())]


def chunk_sections(
    sections: list[tuple[str, str]],
    chunk_size: int = 180,
    overlap: int = 30,
) -> list[dict[str, str]]:
    """Chunk each section independently, retaining its heading in every chunk."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be in [0, chunk_size).")

    chunks: list[dict[str, str]] = []
    step = chunk_size - overlap
    for section, content in sections:
        words = content.split()
        for start in range(0, len(words), step):
            body = " ".join(words[start : start + chunk_size])
            if body:
                chunks.append({"section": section, "text": f"{section}: {body}"})
            if start + chunk_size >= len(words):
                break
    return chunks


def extract_metadata(text: str, path: Path) -> dict[str, Any]:
    """Extract common resume fields using transparent, deterministic heuristics."""
    first_lines = [line.strip() for line in text.splitlines() if line.strip()]
    name = path.stem.replace("_", " ").replace("-", " ").title()
    for line in first_lines[:5]:
        candidate = re.sub(r"^(?:name)\s*:\s*", "", line, flags=re.IGNORECASE).strip()
        if re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,70}", candidate) and len(candidate.split()) <= 5:
            name = candidate
            break

    years = [
        float(value)
        for value in re.findall(r"\b(\d+(?:\.\d+)?)\s*\+?\s+years?\b", text, re.IGNORECASE)
    ]
    education = ""
    for section, content in parse_sections(text):
        if section == "Education":
            education = " ".join(content.split())[:500]
            break

    return {
        "candidate_name": name,
        "skills": ", ".join(extract_skills(text)),
        "experience_years": max(years, default=0.0),
        "education": education or "Not specified",
        "resume_path": str(path),
    }


def read_resume(path: Path) -> str:
    """Read supported plain-text, PDF, and DOCX resumes with explicit dependency errors."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Reading PDF resumes requires pypdf. Install requirements.txt.") from exc
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("Reading DOCX resumes requires python-docx. Install requirements.txt.") from exc
        return "\n".join(paragraph.text for paragraph in Document(str(path)).paragraphs)
    raise ValueError(f"Unsupported resume format: {path.suffix}")


class ResumeRAG:
    """Persist and query section-aware resume embeddings in a local ChromaDB collection."""

    def __init__(
        self,
        persist_directory: str | Path = "vector_db",
        collection_name: str = "resumes",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "ResumeRAG requires ChromaDB and sentence-transformers. Install requirements.txt."
            ) from exc

        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest_directory(
        self,
        resume_directory: str | Path,
        chunk_size: int = 180,
        overlap: int = 30,
    ) -> dict[str, int]:
        """Load all supported resumes beneath a directory and persist their chunks."""
        root = Path(resume_directory)
        if not root.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {root}")
        paths = sorted(
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not paths:
            raise FileNotFoundError(f"No supported resumes found under: {root}")

        ingested_resumes = 0
        ingested_chunks = 0
        for path in paths:
            text = read_resume(path)
            if not text.strip():
                raise ValueError(f"Resume is empty or contains no extractable text: {path}")
            metadata = extract_metadata(text, path)
            chunks = chunk_sections(parse_sections(text), chunk_size, overlap)
            source_key = str(path.resolve())
            self.collection.delete(where={"source_key": source_key})

            ids = []
            chunk_metadata = []
            for index, chunk in enumerate(chunks):
                key = f"{source_key}\0{index}\0{chunk['text']}"
                ids.append(hashlib.sha256(key.encode("utf-8")).hexdigest())
                chunk_metadata.append(
                    {
                        **metadata,
                        "source_key": source_key,
                        "section": chunk["section"],
                        "chunk_index": index,
                    }
                )
            embeddings = self.model.encode(
                [chunk["text"] for chunk in chunks],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=[chunk["text"] for chunk in chunks],
                metadatas=chunk_metadata,
            )
            ingested_resumes += 1
            ingested_chunks += len(chunks)
        return {"resumes": ingested_resumes, "chunks": ingested_chunks}

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Query text cannot be empty.")
        vector = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vector.tolist()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Ingest resume files into a local ChromaDB index.")
    parser.add_argument("--resumes", default="data/resumes", help="Resume directory.")
    parser.add_argument("--db", default="vector_db", help="ChromaDB persistence directory.")
    parser.add_argument("--chunk-size", type=int, default=180)
    parser.add_argument("--overlap", type=int, default=30)
    args = parser.parse_args()
    summary = ResumeRAG(persist_directory=args.db).ingest_directory(
        args.resumes,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(f"Indexed {summary['resumes']} resumes into {summary['chunks']} chunks.")


if __name__ == "__main__":
    main()
