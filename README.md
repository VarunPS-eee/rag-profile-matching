# RAG-Based Profile Matching

A local resume-to-job matching project using section-aware document chunking, Sentence Transformers embeddings, ChromaDB vector storage, and hybrid semantic/skill ranking. The included dataset contains **30 synthetic resumes** and **5 job descriptions**; no real candidate information is used.

## Quick start

Python 3.10+ is recommended. From the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python resume_rag.py --resumes data/resumes
python job_matcher.py --job data/jobs/ml_engineer.txt
python benchmark.py
python -m pytest
```

The first run downloads `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face; allow internet access and a few hundred MB of disk space. ChromaDB and model files are stored locally. No API key is required. Change the model by passing `model_name` when constructing `ResumeRAG`.

## Implementation

- `resume_rag.py` reads `.txt`, `.md`, `.pdf`, and `.docx` resumes, extracts common sections, and chunks each section separately with overlap. It heuristically extracts candidate name, recognized skills, years of experience, and education.
- Sentence Transformers produces normalized embeddings; ChromaDB persists chunk text, metadata, and vectors in `vector_db/`.
- `job_matcher.py` embeds the job description, searches the indexed chunks, aggregates them by resume, enforces explicit must-have skills and experience thresholds, and emits JSON with scores, matched skills, excerpts, and reasoning.
- Hybrid score: 55% best-chunk cosine similarity, 35% coverage of skills named in the job description, and 10% shared-token overlap. Scores are ranking indicators, not calibrated probabilities.
- The matcher queries all indexed chunks before grouping and reranking. This guarantees broad candidate coverage for this small assignment dataset; for a large production corpus, use ANN top-N retrieval plus a separate keyword index.

## Commands

```powershell
python resume_rag.py --resumes data/resumes --db vector_db --chunk-size 180 --overlap 30
python job_matcher.py --job data/jobs/frontend_engineer.txt --top-k 10
python job_matcher.py --job data/jobs/ml_engineer.txt --must-have Python --must-have PyTorch --min-years 5
python benchmark.py
```

Supported resume formats are `.txt`, `.md`, `.pdf`, and `.docx`. PDF and DOCX extraction uses `pypdf` and `python-docx`. The experience-years extractor relies on explicit phrases such as “7 years of experience”; verify extracted metadata before using it in a hiring decision.

## Dataset and evaluation

The resume and job-description files under `data/` are fictional, concise fixtures for demonstrating ingestion and retrieval. `data/evaluation_ground_truth.json` contains curated relevant-name labels. `benchmark.py` reports macro/micro recall@K and mean/p95 end-to-end retrieval latency for those five queries. Results depend on embedding model, hardware, cache state, and corpus size. Synthetic labels and a five-query set are not substitutes for human-reviewed evaluation.

Open `notebooks/rag_experimentation.ipynb` for a reproducible walkthrough, result inspection, and metric discussion. The notebook expects dependencies installed and the local embedding model downloaded.

## Limitations

Metadata and must-have filtering are heuristic and can miss alternate skill names or infer experience inaccurately. The system is a learning project—not a hiring decision tool. Review source resumes, extracted fields, and applicable fairness/privacy requirements before any real-world use.
