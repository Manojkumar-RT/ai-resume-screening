import streamlit as st
import pandas as pd
import pdfplumber
import re
import spacy

nlp = spacy.load("en_core_web_sm")

st.title("AI Resume Screening System")
st.write("Upload resumes and the AI will rank candidates automatically.")

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

    # take only top area of resume
    first_part = text[:2500]

    # remove emails and phones (they confuse AI)
    first_part = re.sub(r'\S+@\S+', ' ', first_part)
    first_part = re.sub(r'(?:\+91[\-\s]?)?[6-9]\d{9}', ' ', first_part)

    doc = nlp(first_part)

    candidates = []

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()

            # filtering garbage detections
            if len(name.split()) >= 2 and len(name) < 40:
                blacklist = [
                    "resume","curriculum","vitae","profile",
                    "objective","declaration","education","project"
                ]

                if not any(b in name.lower() for b in blacklist):
                    candidates.append(name.title())

    if candidates:
        return candidates[0]

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

def decision(score):
    if score >= 8:
        return "Selected"
    elif score >= 5:
        return "Consider"
    else:
        return "Rejected"

if uploaded_files:

    results=[]

    for file in uploaded_files:
        with pdfplumber.open(file) as pdf:
            text=""
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text()

        # -------- IMPORTANT FIX (ADD HERE) --------
        # text cleaning
        text = text.replace("|", " ")
        text = text.replace("•", " ")
        text = re.sub(r'\s+', ' ', text)

        name = extract_name(text)
        email = extract_email(text)
        skills = extract_skills(text)

        phone, experience, education, certifications, projects = extract_details(text)

        # --- NEW SMART SCORING ---
        score = 0

        # skills weight
        if skills and skills != "":
            skill_count = len([s for s in skills.split(",") if s.strip() != ""])
            score += skill_count * 2

        # experience weight
        if experience >= 3:
            score += 5
        elif experience >= 1:
            score += 3

        # education weight
        if "M" in education:
            score += 3
        elif "B" in education:
            score += 2

        # projects
        if projects != "None":
            score += 2

        # certifications
        if certifications != "None":
            score += 2

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

    st.subheader("Candidate Ranking")
    st.dataframe(df)

    st.download_button(
        "Download Results (CSV)",
        df.to_csv(index=False),
        "candidates.csv"
    )