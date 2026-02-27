import streamlit as st
import pandas as pd
import pdfplumber
import re
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return nlp, model

nlp, model = load_models()

st.title("AI Resume Screening System")
st.write("Upload resumes and the AI will rank candidates automatically.")

job_description = st.text_area(
    "Paste Job Description",
    height=200,
    placeholder="Example: Looking for a Data Scientist with Python, Machine Learning, SQL, Pandas..."
)

uploaded_files = st.file_uploader(
    "Upload PDF resumes", accept_multiple_files=True
)

skills_db = [
    "python","java","c++","machine learning","deep learning","sql",
    "html","css","javascript","react","django","flask","nlp","data analysis",
    "pandas","numpy","tableau","power bi"
]

def extract_email(text):
    match = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,4}", text)
    return match[0] if match else "Not Found"

def extract_name(text):

    # Split into lines (IMPORTANT — resumes are line based)
    lines = text.split("\n")

    email = extract_email(text)

    if email == "Not Found":
        return "Not Found"

    # find the line where email appears
    email_index = -1
    for i, line in enumerate(lines):
        if email in line:
            email_index = i
            break

    if email_index == -1:
        return "Not Found"

    # Check 6 lines ABOVE email (THIS is where name actually exists)
    possible_names = lines[max(0, email_index-6):email_index]

    for line in reversed(possible_names):
        line = line.strip()

        # filters
        if (
            5 < len(line) < 40 and
            len(line.split()) >= 2 and
            line.replace(" ", "").isalpha()
        ):
            blacklist = [
                "resume","curriculum","vitae","email","phone",
                "education","skills","projects","objective",
                "profile","declaration","certification"
            ]

            if not any(b in line.lower() for b in blacklist):
                return line.title()

    return "Not Found"

def extract_skills(text):
    text = text.lower()
    found=[]
    for skill in skills_db:
        if skill in text:
            found.append(skill)
    return ", ".join(found)

def extract_details(text):

    # PHONE
    phone_match = re.findall(r'(?:\+91[\-\s]?)?[6-9]\d{9}', text)
    phone = phone_match[0] if phone_match else "Not Found"

    # EXPERIENCE
    # EXPERIENCE (improved)
    experience = 0

    exp_patterns = re.findall(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)', text.lower())
    if exp_patterns:
        experience = int(float(max(exp_patterns)))

    # EDUCATION
    education_keywords = [
        "b.tech","btech","b.e","be","m.tech","mtech","b.sc","bsc",
        "m.sc","msc","bca","mca","phd","mba","bachelor","master"
    ]
    education = "Not Found"
    for word in education_keywords:
        if word in text.lower():
            education = word.upper()
            break

    cert_keywords = ["certification","certified","course","training"]
    certifications = []

    for line in text.split("."):
        if any(k in line.lower() for k in cert_keywords):
            if len(line) > 15 and "@" not in line:
                certifications.append(line.strip())
    certifications = ", ".join(certifications[:3]) if certifications else "None"

    # PROJECTS (improved)
    project_keywords = ["project", "projects"]
    projects = []

    lines = text.split(".")
    for line in lines:
        if any(k in line.lower() for k in project_keywords):
            if len(line) > 20 and len(line) < 200:
                if not re.search(r'\d{10}', line) and "@" not in line:
                    projects.append(line.strip())

    projects = ", ".join(projects[:2]) if projects else "None"

    return phone, experience, education, certifications, projects

def calculate_similarity(resume_text, job_desc):

    if job_desc.strip() == "":
        return 0

    resume_text = resume_text[:3500]
    resume_text = re.sub(r'\S+@\S+', ' ', resume_text)
    resume_text = re.sub(r'(?:\+91[\-\s]?)?[6-9]\d{9}', ' ', resume_text)

    # encode separately
    resume_embedding = model.encode(resume_text)
    job_embedding = model.encode(job_desc)

    similarity = cosine_similarity(
        [resume_embedding],
        [job_embedding]
    )[0][0]

    return round(similarity * 100, 2)

def decision(score):
    if score >= 75:
        return "Strong Match"
    elif score >= 50:
        return "Moderate Match"
    else:
        return "Low Match"

if uploaded_files and job_description.strip() == "":
    st.warning("⚠ Please paste a Job Description first.")
    st.stop()

if uploaded_files:

    results=[]

    for file in uploaded_files:
        with pdfplumber.open(file) as pdf:
            text=""
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if page_text:
                    text += page_text + "\n"

        # -------- IMPORTANT FIX (ADD HERE) --------
        # text cleaning (SAFE CLEANING)
        text = text.replace("|", " ")
        text = text.replace("•", " ")
        text = text.replace("\t", " ")

        name = extract_name(text)
        email = extract_email(text)
        skills = extract_skills(text)

        important_text = text[:2500] + " " + extract_skills(text)
        ml_score = calculate_similarity(important_text, job_description)

        phone, experience, education, certifications, projects = extract_details(text)

        # --- ML SCORE ---
        score = ml_score

        status = decision(score)

        results.append([
            name,
            email,
            phone,
            experience,
            education,
            skills,
            projects,
            certifications,
            score,
            status
        ])

    df = pd.DataFrame(results,columns=[
    "Name",
    "Email",
    "Phone",
    "Experience(Years)",
    "Education",
    "Skills",
    "Projects",
    "Certifications",
    "Score",
    "Status"
])

    st.subheader("Candidate Ranking (Best match at top)")
    df = df.sort_values(by="Score", ascending=False)
    st.dataframe(df)

    # ---- Excel Download ----
    import io
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine='openpyxl')

    st.download_button(
        "Download Results (Excel)",
        buffer.getvalue(),
        "candidates.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )