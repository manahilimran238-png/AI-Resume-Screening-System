
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)


# -----------------------------
# Load Resume Dataset
# -----------------------------

import zipfile
import os

@st.cache_data
def load_data():
    zip_path = "Resume.zip"

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        csv_files = [
            file for file in zip_ref.namelist()
            if file.lower().endswith(".csv")
        ]

        if not csv_files:
            raise FileNotFoundError(
                "No CSV file found inside Resume.zip"
            )

        with zip_ref.open(csv_files[0]) as file:
            return pd.read_csv(file)


resume_data = load_data()


# -----------------------------
# Text Cleaning
# -----------------------------

def clean_text(text):
    return str(text).lower()


# -----------------------------
# Application Title
# -----------------------------

st.title("📄 AI Resume Screening System")

st.write(
    "Enter a job description to find and rank the most relevant "
    "resumes using TF-IDF and Cosine Similarity."
)


# -----------------------------
# Dataset Information
# -----------------------------

st.info(f"📊 {len(resume_data)} resumes available for screening.")


# -----------------------------
# Job Description
# -----------------------------

st.subheader("💼 Job Description")

job_description = st.text_area(
    "Paste the job description below:",
    height=200,
    placeholder=(
        "Example: We are looking for a Data Scientist with "
        "skills in Python, Machine Learning, SQL, and data analysis."
    )
)


# -----------------------------
# Shortlisting Threshold
# -----------------------------

threshold = st.slider(
    "Minimum Match Score for Shortlisting (%)",
    min_value=0,
    max_value=100,
    value=50
)


# -----------------------------
# Screen Resumes
# -----------------------------

if st.button("🔍 Screen Resumes"):

    if not job_description.strip():

        st.warning("Please enter a job description first.")

    else:

        with st.spinner("Analyzing resumes..."):

            # Use the actual Resume_str column
            resume_text = (
                resume_data["Resume_str"]
                .fillna("")
                .apply(clean_text)
            )

            cleaned_job_description = clean_text(
                job_description
            )

            # -----------------------------
            # TF-IDF
            # -----------------------------

            vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words="english"
            )

            resume_vectors = vectorizer.fit_transform(
                resume_text
            )

            job_vector = vectorizer.transform(
                [cleaned_job_description]
            )

            # -----------------------------
            # Cosine Similarity
            # -----------------------------

            similarity_scores = cosine_similarity(
                resume_vectors,
                job_vector
            ).flatten()

            # -----------------------------
            # Create Results
            # -----------------------------

            results = resume_data[
                ["ID", "Category"]
            ].copy()

            results["Match_Score"] = (
                similarity_scores * 100
            ).round(2)

            # Rank candidates
            results = results.sort_values(
                by="Match_Score",
                ascending=False
            ).reset_index(drop=True)

            results["Rank"] = results.index + 1

            # -----------------------------
            # Shortlisting
            # -----------------------------

            shortlisted = results[
                results["Match_Score"] >= threshold
            ]


        st.success("✅ Resume screening completed successfully!")


        # -----------------------------
        # Summary
        # -----------------------------

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Resumes Analyzed",
                len(results)
            )

        with col2:
            st.metric(
                "Shortlisted",
                len(shortlisted)
            )

        with col3:
            st.metric(
                "Highest Match",
                f"{results['Match_Score'].max():.2f}%"
            )


        # -----------------------------
        # Top Candidates
        # -----------------------------

        st.subheader("🏆 Top 10 Candidates")

        top_candidates = results.head(10).copy()

        top_candidates.columns = [
            "Candidate ID",
            "Resume Category",
            "Match Score (%)",
            "Rank"
        ]

        st.dataframe(
            top_candidates[
                [
                    "Rank",
                    "Candidate ID",
                    "Resume Category",
                    "Match Score (%)"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


        # -----------------------------
        # Shortlisted Candidates
        # -----------------------------

        st.subheader("✅ Shortlisted Candidates")

        if len(shortlisted) > 0:

            shortlisted_display = shortlisted.copy()

            shortlisted_display.columns = [
                "Candidate ID",
                "Resume Category",
                "Match Score (%)",
                "Rank"
            ]

            st.dataframe(
                shortlisted_display[
                    [
                        "Rank",
                        "Candidate ID",
                        "Resume Category",
                        "Match Score (%)"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "No candidates reached the selected "
                "shortlisting threshold."
            )


        # -----------------------------
        # Visualization
        # -----------------------------

        st.subheader("📊 Top 10 Resume Match Scores")

        chart_data = (
            results.head(10)
            .set_index("ID")["Match_Score"]
        )

        st.bar_chart(chart_data)


        # -----------------------------
        # Download Results
        # -----------------------------

        csv_data = results.to_csv(
            index=False
        )

        st.download_button(
            label="⬇️ Download Screening Results",
            data=csv_data,
            file_name="resume_screening_results.csv",
            mime="text/csv"
        )
