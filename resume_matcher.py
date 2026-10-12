"""Core resume parsing and matching helpers."""

from __future__ import annotations

import re
from typing import Any


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract selectable text from a PDF, raising a clear error for unusable files."""
    import fitz

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            text = "\n".join(page.get_text() for page in document)
    except (fitz.FileDataError, RuntimeError) as error:
        raise ValueError("This PDF could not be read. Try uploading a valid, text-based PDF.") from error

    if not text.strip():
        raise ValueError(
            "No selectable text was found in this PDF. Scanned resumes need OCR before upload."
        )
    return text


def normalize_text(text: str) -> str:
    """Normalize text for case-insensitive keyword comparisons."""
    return " ".join(
        re.findall(r"[a-z0-9+#]+(?:\.[a-z0-9+#]+)*", text.lower())
    )


def extract_keywords(text: str, nlp: Any) -> list[str]:
    """Extract useful noun phrases and named entities with spaCy."""
    doc = nlp(text)
    candidates: set[str] = set()
    excluded = {
        "ability",
        "company",
        "day",
        "experience",
        "job",
        "level",
        "opportunity",
        "position",
        "responsibility",
        "role",
        "team",
        "work",
        "year",
    }

    for span in (*doc.noun_chunks, *doc.ents):
        phrase = normalize_text(span.text)
        words = phrase.split()
        if (
            phrase
            and len(phrase) > 1
            and len(words) <= 6
            and any(word not in excluded for word in words)
            and not all(word.isdigit() for word in words)
        ):
            candidates.add(phrase)

    return sorted(candidates, key=lambda phrase: (phrase, len(phrase)))


def find_missing_keywords(
    job_keywords: list[str], resume_text: str
) -> tuple[list[str], list[str]]:
    """Split extracted job-description terms into present and missing terms."""
    normalized_resume = f" {normalize_text(resume_text)} "
    matched: list[str] = []
    missing: list[str] = []

    for keyword in job_keywords:
        normalized_keyword = normalize_text(keyword)
        if normalized_keyword and f" {normalized_keyword} " in normalized_resume:
            matched.append(keyword)
        else:
            missing.append(keyword)

    return matched, missing


def cosine_similarity_score(resume_embedding: Any, job_embedding: Any) -> float:
    """Return cosine similarity as a percentage for two embedding vectors."""
    import numpy as np

    resume_vector = np.asarray(resume_embedding, dtype=float).reshape(-1)
    job_vector = np.asarray(job_embedding, dtype=float).reshape(-1)
    if resume_vector.shape != job_vector.shape:
        raise ValueError("Resume and job-description embeddings must have the same shape.")

    denominator = np.linalg.norm(resume_vector) * np.linalg.norm(job_vector)
    if denominator == 0:
        raise ValueError("The embedding model returned an empty vector.")
    return float(np.clip(np.dot(resume_vector, job_vector) / denominator, -1, 1) * 100)


def build_suggestions(missing_keywords: list[str]) -> list[str]:
    """Create actionable resume-edit suggestions from missing job terms."""
    if not missing_keywords:
        return [
            "Your resume includes all of the extracted job-description terms. "
            "Keep the wording accurate and make your strongest evidence easy to scan."
        ]

    suggestions = [
        "If accurate, add evidence for these job-description terms: "
        + ", ".join(f"`{keyword}`" for keyword in missing_keywords[:8])
        + "."
    ]
    suggestions.append(
        "Update the summary or skills section with relevant terminology, then reinforce "
        "it in experience bullets with specific actions and measurable outcomes."
    )
    if len(missing_keywords) > 8:
        suggestions.append(
            f"There are {len(missing_keywords) - 8} additional extracted terms to review; "
            "prioritize the ones most relevant to your actual experience."
        )
    return suggestions
