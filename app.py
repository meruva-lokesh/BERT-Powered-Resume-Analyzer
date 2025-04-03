from flask import Flask, request, render_template, flash, redirect, url_for
from resume_analyzer import extract_text, preprocess_text, calculate_similarity
import re

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Needed for flash messages

# A sample job description (this can be made dynamic later)
JOB_DESCRIPTION = "Looking for a software engineer skilled in Python, machine learning, and NLP."


# Function to find missing keywords and suggest locations
def get_missing_keywords_with_locations(resume_text, job_description):
    # Clean both texts
    clean_resume = preprocess_text(resume_text)
    clean_job = preprocess_text(job_description)

    # Split into words
    resume_words = set(clean_resume.split())
    job_words = set(clean_job.split())

    # Find words in job description that aren’t in resume
    missing_words = job_words - resume_words

    # Filter for important keywords (you can adjust this list)
    important_keywords = {"python", "machine", "learning", "nlp", "software", "engineer"}
    missing_keywords = missing_words & important_keywords  # Intersection of sets

    # Simple rules to suggest where to add them
    suggestions = []
    if missing_keywords:
        for keyword in missing_keywords:
            if keyword in {"python", "nlp"}:  # Programming languages or tools
                suggestions.append(f"Add '{keyword}' to your Skills section.")
            elif keyword in {"machine", "learning"}:  # Concepts or techniques
                suggestions.append(f"Add '{keyword}' to your Skills or Experience section.")
            elif keyword in {"software", "engineer"}:  # Job roles or broad terms
                suggestions.append(f"Add '{keyword}' to your Experience or Summary section.")

    return missing_keywords, suggestions


@app.route("/", methods=["GET", "POST"])
def index():
    uploaded_file = None  # Variable to store the filename
    if request.method == "POST":
        file = request.files["resume"]
        if file:
            # Save the uploaded file temporarily
            file_extension = file.filename.split(".")[-1]
            file_path = "uploaded_resume." + file_extension
            file.save(file_path)

            # Process the resume
            resume_text = extract_text(file_path)
            cleaned_resume_text = preprocess_text(resume_text)
            match_score = calculate_similarity(cleaned_resume_text, JOB_DESCRIPTION)

            # Get missing keywords and location suggestions
            missing_keywords, suggestions = get_missing_keywords_with_locations(resume_text, JOB_DESCRIPTION)
            if missing_keywords:
                advice = "To improve your score, consider these suggestions:\n- " + "\n- ".join(suggestions)
            else:
                advice = "Great job! Your resume matches well."

            # Store the filename for display
            uploaded_file = file.filename

            # Flash the match score and advice
            flash(f"Match Score: {match_score}%")
            flash(advice)
            return render_template("index.html", uploaded_file=uploaded_file)
    return render_template("index.html", uploaded_file=uploaded_file)


if __name__ == "__main__":
    app.run(debug=True)