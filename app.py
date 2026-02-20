import streamlit as st
import pandas as pd
import pdfplumber
import re

st.title("AI Resume Screening System")
st.write("Upload resumes and the AI will rank candidates automatically.")

uploaded_files = st.file_uploader(
    "Upload PDF resumes", accept_multiple_files=True
)

# ---------------- SKILL DATABASE ----------------
skills_db = [
    "python","java","c++","machine learning","deep learning","sql",
    "html","css","javascript","react","django","flask","nlp","data analysis",
    "pandas","numpy","tableau","power bi"
]

# ---------------- EMAIL ----------------
def extract_email(text):
    match = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,4}", text)
    return match[0] if match else "Not Found"

# ---------------- NAME (FIXED AI LOGIC) ----------------
def extract_name(text):

    lines = text.split("\n")

    blacklist = [
        "resume","curriculum vitae","profile","objective","summary",
        "contact","education","skills","projects","experience",
        "declaration","career","personal details"
    ]

    for line in lines[:25]:   # Only check top of resume
        line_clean = line.strip()

        if len(line_clean) < 5:
            continue

        if any(word in line_clean.lower() for word in blacklist):
            continue

        # Name pattern (2–4 words)
        if re.match(r'^[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3}$', line_clean):
            return line_clean

    return "Not Found"

# ---------------- SKILLS ----------------
def extract_skills(text):
    text = text.lower()
    found=[]
    for skill in skills_db:
        if skill in text:
            found.append(skill)
    return ", ".join(found)

# ---------------- DETAILS ----------------
def extract_details(text):

    # PHONE
    phone_match = re.findall(r'(?:\+91[\-\s]?)?[6-9]\d{9}', text)
    phone = phone_match[0] if phone_match else "Not Found"

    # EXPERIENCE (Improved detection)
    experience = 0
    patterns = [
        r'(\d+)\+?\s*years',
        r'(\d+)\+?\s*yrs',
        r'(\d+)\s*year experience',
        r'experience\s*:\s*(\d+)'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            experience = max([int(m) for m in matches])

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

    # CERTIFICATIONS
    cert_keywords = ["certification","certified","course","training"]
    certifications = []
    for line in text.split("\n"):
        if any(k in line.lower() for k in cert_keywords):
            certifications.append(line.strip())
    certifications = ", ".join(certifications) if certifications else "None"

    # PROJECTS
    project_keywords = ["project","projects","developed","built","created"]
    projects = []
    for line in text.split("\n"):
        if any(k in line.lower() for k in project_keywords):
            projects.append(line.strip())
    projects = ", ".join(projects[:3]) if projects else "None"

    return phone, experience, education, certifications, projects

# ---------------- DECISION ----------------
def decision(score):
    if score >= 12:
        return "Selected"
    elif score >= 7:
        return "Consider"
    else:
        return "Rejected"

# ---------------- MAIN PROCESS ----------------
if uploaded_files:

    results=[]

    for file in uploaded_files:
        with pdfplumber.open(file) as pdf:
            text=""
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"

        # Text Cleaning (VERY IMPORTANT)
        text = text.replace("|", " ")
        text = text.replace("•", " ")
        text = re.sub(r'\s+', ' ', text)

        # Extraction
        name = extract_name(text)
        email = extract_email(text)
        skills = extract_skills(text)
        phone, experience, education, certifications, projects = extract_details(text)

        # ---------------- SMART SCORING ----------------
        score = 0

        # skills weight
        if skills:
            skill_count = len([s for s in skills.split(",") if s.strip() != ""])
            score += skill_count * 2

        # experience
        if experience >= 3:
            score += 5
        elif experience >= 1:
            score += 3

        # education
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