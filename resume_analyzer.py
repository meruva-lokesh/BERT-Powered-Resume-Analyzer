import fitz  # PyMuPDF (for PDFs)
import docx  # python-docx (for Word files)
import re
import nltk
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download NLTK stopwords
nltk.download("stopwords")

# Load BERT Model & Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        pdf_document = fitz.open(pdf_path)
        for page in pdf_document:
            text += page.get_text("text") + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"


# Function to extract text from DOCX
def extract_text_from_docx(docx_path):
    text = ""
    try:
        doc = docx.Document(docx_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        return f"Error reading DOCX: {e}"


# Function to extract text based on file type
def extract_text(file_path):
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        return "Unsupported file format. Please upload a PDF or DOCX file."


# Function to clean and preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text


# Function to get BERT embeddings
def get_bert_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Extract [CLS] token embedding


# Function to calculate similarity score
def calculate_similarity(resume_text, job_description):
    resume_embedding = get_bert_embedding(resume_text)
    job_embedding = get_bert_embedding(job_description)
    similarity_score = cosine_similarity(resume_embedding, job_embedding)
    return np.round(similarity_score[0][0] * 100, 2)  # Convert to percentage


# New function to detect sections and analyze keywords
def analyze_resume_keywords(resume_text, job_description):
    # Clean both texts
    clean_resume = preprocess_text(resume_text)
    clean_job = preprocess_text(job_description)

    # Split into words
    resume_words = set(clean_resume.split())
    job_words = set(clean_job.split())

    # Find missing words
    missing_words = job_words - resume_words

    # Define important keywords (expand this for different jobs)
    important_keywords = {
        "skills": {"python", "nlp", "java", "sql", "tensorflow"},  # Technical skills
        "concepts": {"machine", "learning", "data", "analysis"},  # Broader concepts
        "roles": {"software", "engineer", "developer", "manager"}  # Job titles
    }

    # Filter missing keywords by category
    missing_keywords = {}
    for category, keywords in important_keywords.items():
        missing_keywords[category] = missing_words & keywords

    # Try to detect sections in the raw resume text
    sections = {
        "skills": "skills" in resume_text.lower(),
        "experience": "experience" in resume_text.lower(),
        "summary": "summary" in resume_text.lower() or "objective" in resume_text.lower()
    }

    # Generate suggestions based on sections and keywords
    suggestions = []
    if any(missing_keywords.values()):
        for category, keywords in missing_keywords.items():
            if keywords:
                for keyword in keywords:
                    if category == "skills" and sections["skills"]:
                        suggestions.append(f"Add '{keyword}' to your Skills section.")
                    elif category == "skills" and not sections["skills"]:
                        suggestions.append(f"Add a Skills section and include '{keyword}'.")
                    elif category == "concepts" and sections["experience"]:
                        suggestions.append(f"Add '{keyword}' to your Experience section.")
                    elif category == "concepts" and not sections["experience"]:
                        suggestions.append(f"Add an Experience section and include '{keyword}'.")
                    elif category == "roles" and sections["experience"]:
                        suggestions.append(f"Add '{keyword}' to your Experience section.")
                    elif category == "roles" and sections["summary"]:
                        suggestions.append(f"Add '{keyword}' to your Summary section.")
                    else:
                        suggestions.append(f"Add '{keyword}' somewhere in your resume (e.g., Skills or Experience).")
    else:
        suggestions.append("Great job! Your resume matches well.")

    return missing_keywords, suggestions


# Example usage (for testing)
if __name__ == "__main__":
    job_description = "Looking for a software engineer skilled in Python, machine learning, and NLP."
    file_path = "sample_resume.pdf"  # Change this to a real resume file
    resume_text = extract_text(file_path)
    cleaned_resume_text = preprocess_text(resume_text)
    match_score = calculate_similarity(cleaned_resume_text, job_description)

    missing_keywords, suggestions = analyze_resume_keywords(resume_text, job_description)
    print("Resume Match Score:", match_score, "%")
    print("Missing Keywords:", missing_keywords)
    print("Suggestions:")
    for suggestion in suggestions:
        print(f"- {suggestion}")