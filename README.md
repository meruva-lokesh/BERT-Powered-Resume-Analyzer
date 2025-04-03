# AI Resume Analyzer

An AI-powered tool to analyze resumes by extracting text from PDF/DOCX files, preprocessing the content, and comparing it to a job description using BERT embeddings. The project calculates a match score (0-100%) that indicates how well a resume aligns with the job requirements.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Overview

The **AI Resume Analyzer** is built using Python, Flask, and various NLP libraries. It serves as a web application where users can upload their resume (in PDF or DOCX format). The tool then:
1. Extracts text from the uploaded file.
2. Cleans and preprocesses the extracted text.
3. Uses BERT embeddings to compare the resume with a sample job description.
4. Calculates and displays a match score indicating the resume's relevance.

This project is a great addition to a portfolio for those interested in AI, ML, and Natural Language Processing.

## Features

- **File Upload:** Supports PDF and DOCX formats.
- **Text Extraction:** Uses PyMuPDF for PDFs and python-docx for DOCX files.
- **Text Preprocessing:** Cleans and processes resume text (lowercasing, removing extra spaces and stopwords).
- **BERT Embeddings:** Leverages pre-trained BERT models to generate text embeddings.
- **Similarity Calculation:** Uses cosine similarity to compute a match score between the resume and a sample job description.
- **Web Interface:** Simple and user-friendly frontend built with Flask.


