# AI Resume Analyzer (Flask + BERT)

An AI-powered web app that analyzes a resume (PDF/DOCX), compares it to a target job description using BERT embeddings, scores the match, and suggests missing keywords and sections to improve alignment.

## Features
- Upload PDF or DOCX resumes via a clean drag-and-drop UI.
- Extract text from resumes (PyMuPDF for PDFs, python-docx for DOCX).
- Preprocess using NLTK stopwords and regex normalization.
- Compute semantic similarity with the job description using BERT ([CLS] embedding) + cosine similarity.
- Identify missing keywords by category (skills, concepts, roles) and generate actionable suggestions.
- Flash-based feedback with match score and guidance.

## Tech Stack
- Backend: Flask (Python)
- NLP/ML: Hugging Face Transformers (BERT), PyTorch, NLTK, scikit-learn, NumPy
- Document parsing: PyMuPDF (fitz), python-docx
- Frontend: HTML/CSS/JS (Jinja2 templates)

## Project Structure
```text
AI_Resume_Analyzer/
├─ app.py                       # Flask app (routes, file handling, UI rendering)
├─ resume_analyzer.py           # NLP utilities (extraction, preprocessing, BERT similarity, suggestions)
├─ templates/
│  └─ index.html                # Frontend UI
├─ .gitignore                   # Ignores venv, caches, IDE files, uploads, etc.
├─ requirements.txt             # Python dependencies (recommended)
└─ README.md                    # This file
```

## Getting Started

### Prerequisites
- Python 3.9+ recommended
- pip
- (Optional) A GPU-enabled PyTorch installation for faster embedding

### 1) Create and activate a virtual environment
```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
Create a requirements.txt if you don't have one yet:
```txt
Flask
PyMuPDF
python-docx
nltk
transformers
torch
scikit-learn
numpy
```
Then install:
```bash
pip install -r requirements.txt
```

If you prefer ad-hoc installation:
```bash
pip install Flask PyMuPDF python-docx nltk transformers torch scikit-learn numpy
```

### 3) Download NLTK data (stopwords)
The app downloads stopwords at runtime, but to pre-download:
```bash
python -c "import nltk; nltk.download('stopwords')"
```

### 4) Run the app
```bash
# From the project root
python app.py
```
Visit:
```
http://127.0.0.1:5000
```

Upload a resume (PDF/DOCX). You’ll see:
- Match Score: percentage similarity with the job description
- Suggestions: prioritized keyword and section improvements

## Configuration

### Job description
By default, app.py includes a constant:
```python
JOB_DESCRIPTION = "Looking for a software engineer skilled in Python, machine learning, and NLP."
```
Ways to customize:
- Quick test: replace the string in app.py.
- Make it dynamic: add a textarea input to templates/index.html and pass it in the POST request; then read it in app.py (request.form["job_description"]).

### File handling
Current behavior saves the uploaded file temporarily with:
```python
file_path = "uploaded_resume." + file_extension
```
Recommendations:
- Use a dedicated uploads directory (e.g., uploads/) and ensure it’s ignored in .gitignore.
- Use werkzeug.utils.secure_filename for safer filenames.

### Model
- The app uses bert-base-uncased. First run will download the model to your Hugging Face cache (~/.cache/huggingface/).
- To change models, update:
```python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
```

## How It Works

1) Extraction
- PDF: PyMuPDF extracts text per page.
- DOCX: python-docx concatenates paragraph text.

2) Preprocessing
- Lowercasing, whitespace and non-word cleanup, stopword removal (NLTK).

3) Embeddings + Similarity
- Tokenize both resume and job description (truncated to 512 tokens).
- Get BERT last_hidden_state and use the [CLS] embedding.
- Compute cosine similarity and scale to percentage.

4) Keyword Gap Analysis
- Token-level comparison after preprocessing to find missing words from the job description.
- Categorize into skills, concepts, roles.
- Suggest where to add missing keywords based on detected resume sections (Skills, Experience, Summary).

## Production Notes
- Use a production server (e.g., gunicorn or waitress) behind a reverse proxy.
- Consider enabling GPU-enabled PyTorch in production for speed.
- Sanitize file uploads, enforce size limits, and consider scanning uploads if exposed publicly.
- Set a strong Flask secret key via environment variable:
```bash
$env:FLASK_SECRET="a-strong-random-secret"  # PowerShell
export FLASK_SECRET="a-strong-random-secret" # bash
```
Then in app.py:
```python
import os
app.secret_key = os.getenv("FLASK_SECRET", "dev_only_secret")
```

## Troubleshooting

- PyMuPDF install issues on Windows:
  - Ensure you install PyMuPDF (not fitz). The package name is `PyMuPDF`.
- Torch installation errors:
  - Use the official selector: https://pytorch.org/get-started/locally/
- NLTK stopwords error:
  - Run `python -c "import nltk; nltk.download('stopwords')"`
- Model download failures:
  - Check internet access and Hugging Face mirrors; set HF_HOME if needed.
- No changes to commit (git):
  - Ensure your edited files are saved and not ignored by .gitignore.

## Roadmap
- Make job description user-provided in UI.
- Extract skills using entity/phrase chunking for better keyword mapping.
- Support more file types (TXT, RTF).
- Per-section scoring (Skills, Experience, Summary).
- Caching embeddings and rate limiting for multi-user deployments.

## Security & Privacy
- Uploaded files are processed on the server. Avoid storing resumes long-term.
- Add automatic cleanup of temporary files.
- Do not log raw resume content in production logs.

## Contributing
- Fork the repo, create a feature branch, and open a pull request.
- Use conventional commits (feat:, fix:, chore:).

## License
This project is provided without a license by default. Add a LICENSE file (e.g., MIT) to clarify usage terms for your repository.

---
Made by an AIML engineer: leveraging BERT-based semantic matching and keyword gap analysis to help candidates tailor their resumes.
