from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example job roles and required keywords
roles = {
    "Data Scientist": "python machine learning pandas numpy statistics regression sklearn",
    "Frontend Developer": "html css javascript react frontend user interface",
    "Backend Developer": "python django flask apis backend database sql",
    "AI/ML Engineer": "deep learning tensorflow pytorch keras machine learning neural networks",
    "DevOps Engineer": "docker kubernetes aws azure jenkins devops automation pipelines"
}

def match_jobs(resume_text):
    vectorizer = TfidfVectorizer()
    all_roles = list(roles.values())
    role_names = list(roles.keys())

    tfidf_matrix = vectorizer.fit_transform([resume_text] + all_roles)
    resume_vector = tfidf_matrix[0]
    role_vectors = tfidf_matrix[1:]

    similarity_scores = cosine_similarity(resume_vector, role_vectors)[0]
    best_index = similarity_scores.argmax()

    best_role = role_names[best_index]
    score = round(similarity_scores[best_index], 2)

    # Get keywords in best role that are missing
    resume_words = set(resume_text.split())
    required_keywords = set(roles[best_role].split())
    missing_keywords = list(required_keywords - resume_words)

    return best_role, score, missing_keywords
