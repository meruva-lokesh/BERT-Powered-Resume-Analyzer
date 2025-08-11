from flask import Flask, request, render_template, flash, redirect, url_for
from resume_analyzer import extract_text, preprocess_text, calculate_similarity, analyze_resume_keywords

app = Flask(__name__)
app.secret_key = "your_secret_key"

# A sample job description (this can be made dynamic later)
JOB_DESCRIPTION = "Looking for a software engineer skilled in Python, machine learning, and NLP."

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

            # Get missing keywords and suggestions
            _, suggestions = analyze_resume_keywords(resume_text, JOB_DESCRIPTION)
            advice = "To improve your score, consider these suggestions:\n- " + "\n- ".join(suggestions) if suggestions[0] != "Great job! Your resume matches well." else suggestions[0]

            # Store the filename for display
            uploaded_file = file.filename

            # Flash the uploaded file info, match score, and advice
            flash(f"You have uploaded: {uploaded_file}")
            flash(f"Match Score: {match_score}%")
            flash(advice)
            return render_template("index.html", uploaded_file=uploaded_file)
    return render_template("index.html", uploaded_file=uploaded_file)

if __name__ == "__main__":
    app.run(debug=True)