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


# Example usage (for testing)
if __name__ == "__main__":
    job_description = "Looking for a software engineer skilled in Python, machine learning, and NLP."
    file_path = "sample_resume.pdf"  # Change this to a real resume file
    resume_text = extract_text(file_path)
    cleaned_resume_text = preprocess_text(resume_text)
    match_score = calculate_similarity(cleaned_resume_text, job_description)

    print("Resume Match Score:", match_score, "%")
