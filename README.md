# AI Resume Matcher & Optimizer

A local Streamlit app that compares a resume PDF with a specific job description.
It reports semantic similarity, highlights extracted job-description terms that
do not appear verbatim in the resume, and suggests truthful ways to tailor the
resume.

## Setup

Use Python 3.10 or newer. From this folder, create and activate a virtual
environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Deploy on Render

The included `render.yaml` configures a free Render web service. Push this
project to a GitHub, GitLab, or Bitbucket repository, then create a Blueprint
on Render and select that repository. Render builds the service and provides
its public `onrender.com` URL when deployment finishes. The first build installs
the Python packages and spaCy language model; the sentence-transformer model
downloads the first time someone analyzes a resume.

Free Render services can take a little time to wake after inactivity. Resume
PDFs are processed in memory and are not stored by the app.

Start the app:

```powershell
streamlit run app.py
```

The first analysis downloads the `all-MiniLM-L6-v2` sentence-transformer model
if it is not already cached. Internet access may be required on first run.

## Using the app

1. Upload a text-based PDF resume.
2. Paste the job description and select **Analyze match**.
3. Review the similarity score, matched terms, terms to review, and suggestions.
4. Optionally download the analysis as a text file.

Scanned/image-only PDFs are not OCR'd. Similarity is a comparison of document
text, and extracted terms are lexical indicators rather than a hiring score.
Terms flagged as missing may be described using different wording, so review
the context before editing. Do not add skills or experience that are not true.

## Tests

Run the focused core tests with:

```powershell
python -m unittest discover -s tests -v
```
