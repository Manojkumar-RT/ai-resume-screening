import streamlit as st
import pandas as pd
import pdfplumber
import spacy
import re

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
    lines = text.split('\n')
    for i,line in enumerate(lines):
        if "@" in line:
            for j in range(max(0,i-5), i):
                name = lines[j].strip()
                if len(name) > 3 and name.replace(" ","").isalpha():
                    return name.title()
    return "Not Found"

def extract_skills(text):
    text = text.lower()
    found=[]
    for skill in skills_db:
        if skill in text:
            found.append(skill)
    return ", ".join(found)

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

        name = extract_name(text)
        email = extract_email(text)
        skills = extract_skills(text)
        score = len(skills.split(",")) if skills else 0
        status = decision(score)

        results.append([name,email,skills,score,status])

    df = pd.DataFrame(results,columns=["Name","Email","Skills","Score","Status"])

    st.subheader("Candidate Ranking")
    st.dataframe(df)

    st.download_button(
        "Download Results (CSV)",
        df.to_csv(index=False),
        "candidates.csv"
    )