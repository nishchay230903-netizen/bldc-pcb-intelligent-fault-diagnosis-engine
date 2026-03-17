import json
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# LOAD DATA
# =========================
with open("pcb_data.json", "r") as f:
    data = json.load(f)

# =========================
# PREPARE TEXT DATA
# =========================
documents = []
faults = []

for entry in data:
    text = " ".join(entry["symptoms"] + entry["causes"])
    documents.append(text)
    faults.append(entry)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# =========================
# DIAGNOSIS FUNCTION
# =========================
def diagnose(query):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, X)[0]

    results = []
    for i, score in enumerate(similarities):
        results.append({
            "fault": faults[i]["fault"],
            "confidence": round(float(score), 2),
            "data": faults[i]
        })

    # sort by similarity
    results = sorted(results, key=lambda x: x["confidence"], reverse=True)

    top = results[0]

    # structured output
    diagnosis = f"""
Primary Fault: {top['fault']}

Possible Causes:
- {'\\n- '.join(top['data']['causes'])}

Recommended Tests:
- {'\\n- '.join(top['data']['tests'])}

Fix:
- {'\\n- '.join(top['data']['fix'])}
"""

    return diagnosis, results[:5]

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="BLDC AI Diagnostic Engine", layout="wide")
st.title("⚡ BLDC PCB AI Diagnostic Engine (Offline Version)")

user_input = st.text_area("Describe PCB issue:")

if st.button("Run Diagnosis"):
    if user_input:
        result, ranked = diagnose(user_input)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🧠 Diagnosis")
            st.text(result)

        with col2:
            st.subheader("📊 Fault Confidence Ranking")
            faults = [f["fault"] for f in ranked]
            scores = [f["confidence"] for f in ranked]

            fig = plt.figure()
            plt.barh(faults, scores)
            plt.xlabel("Confidence")
            plt.ylabel("Fault")
            plt.title("Fault Similarity Ranking")
            st.pyplot(fig)

    else:
        st.warning("Enter a valid issue")