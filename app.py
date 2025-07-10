import streamlit as st
from resume_parser import extract_text_from_pdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Streamlit Page Settings ---
st.set_page_config(page_title="🧠 AI Resume Analyzer", layout="centered")
st.title("🧠 AI Resume vs Job Description Matcher")
st.markdown("Compare your resume with a job description and get a match score plus missing keywords!")

# --- File Uploads ---
resume_file = st.file_uploader("📄 Upload your Resume (PDF only)", type=["pdf"])
jd_file = st.file_uploader("📑 Upload Job Description (PDF only)", type=["pdf"])

# --- When both files are uploaded ---
if resume_file and jd_file:
    # Extract text
    resume_text = extract_text_from_pdf(resume_file)
    jd_text = extract_text_from_pdf(jd_file)

    # Vectorize and compute similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    score_percent = round(score * 100, 2)

    # Display Score
    st.subheader("🔍 Match Result:")
    st.success(f"**Match Score: {score_percent}/100**")

    # Analyze keywords
    resume_words = set(resume_text.lower().split())
    jd_keywords = set(jd_text.lower().split())
    missing_keywords = list(jd_keywords - resume_words)

    if missing_keywords:
        st.warning("⚠️ Missing Important Keywords from Resume:")
        st.write(", ".join(sorted(missing_keywords)[:20]))  # Limit to top 20
    else:
        st.success("✅ Awesome! Your resume covers all key terms from the job description.")

# --- If only one file uploaded ---
elif resume_file and not jd_file:
    st.info("📑 Please upload a job description to compare.")

elif jd_file and not resume_file:
    st.info("📄 Please upload a resume to compare.")

# --- If nothing uploaded ---
else:
    st.info("⬆️ Upload both resume and job description PDFs to begin.")



