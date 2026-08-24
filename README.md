# 📄 AI Resume Screening System

An AI-based resume screening system that analyzes resumes against a user-provided job description and ranks candidates based on their relevance.

## 📌 Project Overview

Recruiters often receive a large number of resumes for a single job position, making manual screening time-consuming. This project automates the initial screening process by comparing resume content with a job description.

The system uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert resume and job-description text into numerical vectors and **Cosine Similarity** to calculate how closely each resume matches the job requirements.

A **Streamlit web application** allows users to enter a job description and instantly view ranked candidates and shortlisted resumes.

## 🎯 Objectives

* Automate the initial resume screening process
* Compare resumes with a user-provided job description
* Calculate resume-job similarity scores
* Rank candidates based on relevance
* Shortlist candidates using a customizable match-score threshold
* Provide an interactive web interface for recruiters

## 📂 Dataset

The project uses a resume dataset containing different resume categories.

Main columns include:

* `ID` — Unique resume identifier
* `Resume_str` — Resume text
* `Resume_html` — HTML representation of the resume
* `Category` — Resume category

The `Resume_str` column is used for text-based resume matching.

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TF-IDF Vectorization
* Cosine Similarity
* Streamlit
* Matplotlib / Streamlit Charts
* Google Colab / Jupyter Notebook

## 🔄 Project Workflow

```text
Resume Dataset
       ↓
Data Cleaning
       ↓
Resume Text Extraction
       ↓
User Enters Job Description
       ↓
TF-IDF Vectorization
       ↓
Cosine Similarity
       ↓
Calculate Match Scores
       ↓
Rank Resumes
       ↓
Apply Shortlisting Threshold
       ↓
Display Results
```

## 🧠 Methodology

### 1. Text Preprocessing

Resume text and the job description are converted to lowercase and missing resume text values are handled.

### 2. TF-IDF Vectorization

TF-IDF converts the resume text and job description into numerical representations based on the importance of words within the documents.

### 3. Cosine Similarity

Cosine similarity measures the similarity between the job description and each resume.

A higher similarity score indicates that the resume contains more text information relevant to the job description.

### 4. Candidate Ranking

Candidates are sorted from the highest similarity score to the lowest.

### 5. Shortlisting

Users can select a minimum match-score threshold. Resumes meeting or exceeding the threshold are displayed as shortlisted candidates.

## 🖥️ Streamlit Application

The Streamlit application provides:

* User-entered job description
* Adjustable shortlisting threshold
* Number of resumes analyzed
* Number of shortlisted candidates
* Highest matching score
* Top 10 candidate rankings
* Resume categories
* Match-score visualization
* Downloadable screening results

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/manahilimran238-png/AI-Resume-Screening-System.git
cd AI-Resume-Screening-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 Example Job Description

The application can be tested with a Data Scientist job description containing requirements such as:

* Python
* Pandas
* NumPy
* Scikit-learn
* Machine Learning
* SQL
* Data Analysis
* Data Visualization
* Statistics
* Exploratory Data Analysis

The system will compare the job description against the available resumes and rank candidates according to their similarity scores.

## 📈 Output

The system produces:

* Candidate ranking
* Resume category
* Match score
* Shortlisted candidates
* Match-score visualization
* CSV file containing screening results

## ⚠️ Limitations

This system uses text similarity rather than a trained hiring classifier. Therefore, the match score represents **similarity between the job description and resume text**, not the candidate's actual probability of being hired.

The system may also be affected by differences in terminology, formatting, and wording between resumes and job descriptions.



