"""Streamlit UI for comparing a resume with a job description."""

from __future__ import annotations

import streamlit as st
from sentence_transformers import SentenceTransformer
import spacy

from resume_matcher import (
    build_suggestions,
    cosine_similarity_score,
    extract_keywords,
    extract_pdf_text,
    find_missing_keywords,
)


MODEL_NAME = "all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"


@st.cache_resource(show_spinner="Loading semantic matching model...")
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource(show_spinner="Loading spaCy language model...")
def load_spacy_model():
    try:
        return spacy.load(SPACY_MODEL)
    except OSError as error:
        raise RuntimeError(
            "The spaCy English model is missing. Install it with "
            "`python -m spacy download en_core_web_sm`, then restart the app."
        ) from error


@st.cache_data(show_spinner=False)
def analyze_documents(resume_text: str, job_description: str):
    """Cache completed analyses so reruns with the same documents are instant."""
    nlp = load_spacy_model()
    embedding_model = load_embedding_model()
    job_keywords = extract_keywords(job_description, nlp)
    matched_keywords, missing_keywords = find_missing_keywords(
        job_keywords, resume_text
    )
    embeddings = embedding_model.encode(
        [resume_text, job_description],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    # The vectors are normalized by encode(), so their dot product is cosine similarity.
    score = cosine_similarity_score(embeddings[0], embeddings[1])
    return score, matched_keywords, missing_keywords


def create_report(score: float, matched: list[str], missing: list[str]) -> str:
    lines = [
        "AI Resume Matcher - Analysis",
        f"Semantic similarity: {score:.1f}%",
        "",
        f"Matched job-description terms ({len(matched)}):",
        *(f"- {term}" for term in matched),
        "",
        f"Terms to review ({len(missing)}):",
        *(f"- {term}" for term in missing),
        "",
        "Suggestions:",
        *(f"- {suggestion}" for suggestion in build_suggestions(missing)),
        "",
        "Note: Similarity is a text-level estimate, not a hiring recommendation.",
    ]
    return "\n".join(lines)


st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root { --ink:#17243b; --muted:#52627a; --blue:#2949c7; --teal:#087f78; --line:#d5deed; --surface:#ffffff; }
    .stApp {
        background: radial-gradient(ellipse at 8% 4%,rgba(69,105,230,.13),transparent 34%),
                    radial-gradient(ellipse at 96% 14%,rgba(19,168,158,.11),transparent 30%),#f5f7fc;
        color:var(--ink); font-family:Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif;
    }
    .block-container { max-width:1180px; padding-top:2.5rem; padding-bottom:3rem; }
    h1,h2,h3 { color:#14213b!important; font-family:Inter, ui-sans-serif, system-ui, 'Segoe UI', sans-serif; letter-spacing:-.035em; }
    h1 { font-size:clamp(2rem,4vw,3rem)!important; font-weight:800!important; }
    h2,h3 { font-weight:700!important; }
    [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
    label, .stMarkdown, [data-testid="stMarkdownContainer"] p { color:var(--muted); }
    [data-testid="stSidebar"] { background:#eef2fb; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] [data-testid="stAlert"], [data-testid="stAlert"] { border-radius:14px; }
    [data-testid="stFileUploader"], [data-testid="stTextArea"] {
        background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:14px;
        box-shadow:0 8px 28px rgba(29,48,86,.055); transition:border-color .2s,box-shadow .2s;
    }
    [data-testid="stFileUploader"]:hover, [data-testid="stTextArea"]:focus-within {
        border-color:#9badf0; box-shadow:0 10px 30px rgba(52,87,213,.11);
    }
    [data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input {
        background:#fff!important; color:#17243b!important; caret-color:#2949c7;
        border-color:#c8d3e5!important;
    }
    [data-testid="stTextArea"] textarea::placeholder, [data-testid="stTextInput"] input::placeholder { color:#718096!important; opacity:1; }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] { background:#f8faff; }
    [data-testid="stFileUploader"] button { color:#2949c7!important; border-color:#b9c8ea!important; }
    [data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span { color:#45546b; }
    [data-testid="stWidgetLabel"] p { color:#263750!important; font-weight:650; }
    [data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
        border-radius:11px; padding:.62rem 1.2rem; font-weight:700; transition:transform .15s,box-shadow .15s;
    }
    [data-testid="stButton"] button[kind="primary"] {
        background:linear-gradient(115deg,var(--blue),#536fe0); border:0;
        box-shadow:0 7px 18px rgba(52,87,213,.23);
    }
    [data-testid="stButton"] button:disabled {
        opacity:.78!important; color:#fff!important; -webkit-text-fill-color:#fff!important;
        background:#5369bf!important;
    }
    [data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover {
        transform:translateY(-1px); box-shadow:0 9px 20px rgba(29,48,86,.15);
    }
    [data-testid="stMetric"] {
        background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:18px 20px;
        box-shadow:0 8px 28px rgba(29,48,86,.055);
    }
    [data-testid="stMetricLabel"] { color:var(--muted); font-weight:600; }
    [data-testid="stMetricValue"] { color:var(--blue); font-family:Inter, ui-sans-serif, system-ui, 'Segoe UI', sans-serif; font-weight:800; }
    [data-testid="stProgress"] > div > div { background:linear-gradient(90deg,var(--blue),var(--teal)); border-radius:99px; }
    [data-testid="stTabs"] [role="tab"] { font-weight:600; }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] { color:var(--blue); }
    @media (max-width:700px) { .block-container { padding:1.4rem 1rem 2rem; } [data-testid="stMetric"] { padding:14px; } }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AI Resume Matcher & Optimizer")
st.caption(
    "Compare a text-based PDF resume with a specific job description, "
    "review relevant terms, and find opportunities to tailor your resume."
)

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "The semantic score compares the meaning of both documents. "
        "The term checklist highlights phrases found in the job description "
        "that may be absent from your resume."
    )
    st.info(
        "Use the results as editing guidance, not as a hiring prediction. "
        "Only add skills or experience that accurately reflect your background."
    )

left, right = st.columns(2)
with left:
    resume_file = st.file_uploader(
        "Upload your resume (PDF)",
        type=["pdf"],
        help="Text-based PDFs are supported; scanned documents need OCR first.",
    )
with right:
    job_description = st.text_area(
        "Paste the job description",
        height=220,
        placeholder="Include the responsibilities and qualifications for the role...",
    )

analyze = st.button(
    "Analyze match",
    type="primary",
    disabled=resume_file is None or not job_description.strip(),
)

if analyze and resume_file is not None:
    try:
        resume_text = extract_pdf_text(resume_file.getvalue())
    except ValueError as error:
        st.error(str(error))
    else:
        try:
            with st.spinner("Analyzing resume and job description..."):
                score, matched_keywords, missing_keywords = analyze_documents(
                    resume_text, job_description
                )
        except (RuntimeError, OSError, ValueError) as error:
            st.error(f"Analysis could not be completed: {error}")
        else:
            st.subheader("Your match overview")
            metric_columns = st.columns(3)
            metric_columns[0].metric("Semantic similarity", f"{score:.1f}%")
            metric_columns[1].metric("Matched terms", len(matched_keywords))
            metric_columns[2].metric("Terms to review", len(missing_keywords))
            st.progress(max(0.0, min(score / 100, 1.0)))
            st.caption(
                "The score measures semantic similarity between the full texts; "
                "it is not a probability of getting an interview."
            )

            matched_tab, missing_tab, suggestions_tab = st.tabs(
                ["Matched terms", "Missing / review", "Tailoring suggestions"]
            )
            with matched_tab:
                if matched_keywords:
                    st.write("Terms detected in both the job description and resume:")
                    st.write(", ".join(f"`{term}`" for term in matched_keywords))
                else:
                    st.write("No extracted job-description terms were matched exactly.")
            with missing_tab:
                if missing_keywords:
                    st.write(
                        "These extracted phrases were not found exactly in the resume. "
                        "Review them for relevance; related wording may still demonstrate the skill."
                    )
                    st.write(", ".join(f"`{term}`" for term in missing_keywords))
                else:
                    st.success("All extracted job-description phrases appear in the resume.")
            with suggestions_tab:
                for suggestion in build_suggestions(missing_keywords):
                    st.markdown(f"- {suggestion}")

            st.download_button(
                "Download analysis",
                data=create_report(score, matched_keywords, missing_keywords),
                file_name="resume_match_analysis.txt",
                mime="text/plain",
            )
