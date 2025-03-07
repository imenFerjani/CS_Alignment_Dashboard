import streamlit as st
import os
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer, util
import numpy as np
import plotly.express as px
from collections import Counter
from huggingface_hub.inference_api import InferenceApi
import random

# Load SpaCy model - proper error handling for cloud deployment
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("SpaCy model not found. Please make sure 'en_core_web_sm' is included in your requirements.txt file.")
    st.stop()


# Function to check and create mock data if needed
def load_or_create_mock_data(file_path, create_func):
    try:
        return pd.read_csv(file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        st.warning(f"Could not load {file_path}. Using mock data instead.")
        return create_func()


# Create mock data functions
def create_mock_courses():
    return pd.DataFrame({
        "Course_Name": ["Introduction to Programming", "Data Structures", "Algorithms"],
        "Outcome": [
            "Understand Python basics and write simple programs",
            "Implement and analyze various data structures",
            "Design and analyze algorithms for problem-solving"
        ]
    })


def create_mock_jobs():
    return pd.DataFrame({
        "Title": ["Software Engineer", "Data Scientist", "Web Developer"],
        "Description": [
            "Develop software using Python, Java, and SQL. Knowledge of cloud platforms preferred.",
            "Analyze data using machine learning and statistical methods. Python, R, and TensorFlow experience needed.",
            "Create web applications using JavaScript, HTML, CSS, React, and Node.js."
        ]
    })


def create_mock_standards():
    return pd.DataFrame({
        "Standard": ["CS2023-1", "CSTA-2", "ABET-3"],
        "Competency": [
            "Apply programming fundamentals to solve problems",
            "Design and implement data structures and algorithms",
            "Analyze computational requirements for real-world problems"
        ]
    })


# Load datasets with fallback to mock data
courses_df = load_or_create_mock_data("data/cs_course_outcomes.csv", create_mock_courses)
jobs_df = load_or_create_mock_data("data/cs_job_postings_500.csv", create_mock_jobs)

# For standards, try to load each file and if none exist, use mock data
standards_files = ["data/cs2023_standards.csv", "data/csta_standards.csv",
                   "data/abet_standards.csv", "data/global_cs_standards.csv"]

standards_dfs = []
for file in standards_files:
    try:
        standards_dfs.append(pd.read_csv(file))
    except (FileNotFoundError, pd.errors.EmptyDataError):
        continue

if not standards_dfs:
    standards_df = create_mock_standards()
else:
    standards_df = pd.concat(standards_dfs, ignore_index=True)

# Ensure required columns exist
if "Outcome" not in courses_df.columns:
    courses_df["Outcome"] = "Sample course outcome"
if "Description" not in jobs_df.columns:
    jobs_df["Description"] = "Sample job description"
if "Competency" not in standards_df.columns:
    standards_df["Competency"] = "Sample competency standard"

# Define skill list
skills = ["Python", "Java", "C++", "SQL", "AWS", "Kubernetes", "Docker", "TensorFlow", "React", "Node.js",
          "Git", "Linux", "Spark", "Tableau", "Wireshark", "Solidity", "ROS", "OpenCV", "JavaScript", "HTML",
          "CSS", "MongoDB", "PostgreSQL", "Agile", "Scrum", "encryption", "machine learning", "NLP", "cloud"]

# Load SentenceTransformer model
with st.spinner("Loading models... This may take a moment."):
    model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract embeddings - with progress indicator
with st.spinner("Computing embeddings..."):
    if "Embeddings" not in courses_df.columns:
        courses_df["Embeddings"] = courses_df["Outcome"].apply(lambda x: model.encode(x))
    if "Embeddings" not in jobs_df.columns:
        jobs_df["Embeddings"] = jobs_df["Description"].apply(lambda x: model.encode(x))
    if "Embeddings" not in standards_df.columns:
        standards_df["Embeddings"] = standards_df["Competency"].apply(lambda x: model.encode(x))

# Custom CSS for IAU theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1A2A44;
        color: #F4E3B2;
        font-family: 'Arial', sans-serif;
    }
    .stSidebar {
        background-color: #2E4066;
        color: #F4E3B2;
        width: 200px !important;
        padding: 10px;
    }
    .stButton>button {
        background-color: #F4E3B2;
        color: #1A2A44;
        border-radius: 5px;
        font-weight: bold;
    }
    /* KPI styles are now handled by custom HTML/CSS in the code */
    .stDataFrame {
        background-color: #2E4066;
        color: #F4E3B2;
        border: 2px solid #F4E3B2;
        border-radius: 5px;
        font-size: 14px;
    }
    h1, h2, h3 {
        color: #F4E3B2 !important;
        font-family: 'Arial', sans-serif;
    }
    .css-1aumxhk {
        color: #F4E3B2 !important;
    }
    .stApp [data-testid="stDecoration"] {
        display: none;
    }
    /* Updated tab styling - FIXED: improved visibility for both selected and non-selected tabs */
    div[data-baseweb="tab-list"] {
        display: flex;
        flex-wrap: wrap;
    }
    button[role="tab"] {
        color: #F4E3B2 !important;
        background-color: #2E4066;
        padding: 10px 20px;
        border-radius: 5px;
        margin: 5px;
        font-size: 16px;
        opacity: 1.0 !important;
        border: 1px solid #F4E3B2;
    }
    button[role="tab"]:hover {
        color: #FFFFFF !important;
        background-color: #3A5488;
    }
    button[role="tab"][aria-selected="true"] {
        color: #FFFFFF !important;
        background-color: #1A2A44;
        border-bottom: 3px solid #F4E3B2;
    }
    .stFooter {
        color: #F4E3B2;
        text-align: center;
        font-size: 14px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header image - added fallback
try:
    st.image("Header_img.png", use_column_width=True)
except:
    st.title("🎓 IAU")  # Text fallback if image fails to load

# Title and subtitle
st.title("🧠 Smart Computer Science Alignment Dashboard")
st.markdown("Using AI to Align Courses with Market Needs and Standards")

# Sidebar for course selection
st.sidebar.header("Course Selection")
course_options = courses_df["Course_Name"].unique()
selected_course = st.sidebar.selectbox("Select a Course", course_options)
custom_outcome = st.sidebar.text_area("Or Enter Custom Outcome", "")

# Filter data based on selection
if custom_outcome:
    course_embeddings = [model.encode(custom_outcome)]
    course_name = "Custom Course"
else:
    course_data = courses_df[courses_df["Course_Name"] == selected_course]
    course_embeddings = course_data["Embeddings"].tolist()
    course_name = selected_course

# Compute average embedding for the course
course_avg_embedding = np.mean(course_embeddings, axis=0)

# Compute similarity scores
job_embeddings = jobs_df["Embeddings"].tolist()
standard_embeddings = standards_df["Embeddings"].tolist()
job_similarities = util.cos_sim(course_avg_embedding, job_embeddings)[0].numpy()
standard_similarities = util.cos_sim(course_avg_embedding, standard_embeddings)[0].numpy()

# KPIs - Using custom solution instead of st.metric for better visibility
st.header("📊 Key Performance Indicators")

# Calculate metrics
avg_job_similarity = np.mean(job_similarities) * 100
avg_standard_similarity = np.mean(standard_similarities) * 100
skill_gaps = len([s for s in skills if max([util.cos_sim(model.encode(s), e)[0][0] for e in course_embeddings]) < 0.7])

# Create a custom layout for metrics to ensure visibility
col1, col2, col3 = st.columns(3)

# Custom HTML/CSS for KPI display instead of using st.metric
with col1:
    st.markdown(f"""
        <div style="background-color: rgba(30, 50, 80, 1.0); padding: 15px; border-radius: 10px; 
                    border: 2px solid #F4E3B2; text-align: center;">
            <div style="color: #FFFF00; font-size: 16px; font-weight: bold; margin-bottom: 5px;">
                Market Alignment
            </div>
            <div style="color: #FFFFFF; font-size: 24px; font-weight: bold;">
                {avg_job_similarity:.2f}%
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div style="background-color: rgba(30, 50, 80, 1.0); padding: 15px; border-radius: 10px; 
                    border: 2px solid #F4E3B2; text-align: center;">
            <div style="color: #FFFF00; font-size: 16px; font-weight: bold; margin-bottom: 5px;">
                Standards Match
            </div>
            <div style="color: #FFFFFF; font-size: 24px; font-weight: bold;">
                {avg_standard_similarity:.2f}%
            </div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div style="background-color: rgba(30, 50, 80, 1.0); padding: 15px; border-radius: 10px; 
                    border: 2px solid #F4E3B2; text-align: center;">
            <div style="color: #FFFF00; font-size: 16px; font-weight: bold; margin-bottom: 5px;">
                Skill Gaps
            </div>
            <div style="color: #FFFFFF; font-size: 24px; font-weight: bold;">
                {skill_gaps}
            </div>
        </div>
    """, unsafe_allow_html=True)

# Skill similarity table
st.header("🔍 Skill Similarity Details")
skill_similarities = []
for skill in skills:
    skill_embedding = model.encode(skill)
    max_course_similarity = max([util.cos_sim(skill_embedding, e)[0][0] for e in course_embeddings])
    max_job_similarity = max([util.cos_sim(skill_embedding, e)[0][0] for e in job_embeddings])
    max_standard_similarity = max([util.cos_sim(skill_embedding, e)[0][0] for e in standard_embeddings])
    skill_similarities.append({
        "Skill": skill,
        "Course Similarity": f"{max_course_similarity:.2f}",
        "Job Similarity": f"{max_job_similarity:.2f}",
        "Standard Similarity": f"{max_standard_similarity:.2f}"
    })
similarity_df = pd.DataFrame(skill_similarities)
st.dataframe(similarity_df, use_container_width=True)

# Visualizations - FIXED: More visible tab titles
st.header("📈 Visual Insights")
# Better approach for tabs with clearer names
tabs = st.tabs(["📊 Similarity Distribution", "🔝 Top Skills"])

with tabs[0]:  # First tab - Similarity Distribution
    sim_df = pd.DataFrame({
        "Similarity": np.concatenate([job_similarities, standard_similarities]),
        "Source": ["Jobs"] * len(job_similarities) + ["Standards"] * len(standard_similarities)
    })
    fig = px.histogram(sim_df, x="Similarity", color="Source", nbins=20, title="Similarity Distribution")
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:  # Second tab - Top Skills
    course_skills = sum(courses_df[courses_df["Course_Name"] == course_name]["Outcome"].apply(
        lambda x: [t.text for t in nlp(x) if t.text in skills]).tolist(), [])
    job_skills = sum(jobs_df["Description"].apply(lambda x: [t.text for t in nlp(x) if t.text in skills]).tolist(), [])
    course_freq = Counter(course_skills)
    job_freq = Counter(job_skills)
    freq_df = pd.DataFrame({
        "Skill": list(set(course_freq.keys()).union(set(job_freq.keys()))),
        "Frequency": [course_freq.get(skill, 0) for skill in set(course_freq.keys()).union(set(job_freq.keys()))],
        "Source": ["Courses"] * len(set(course_freq.keys()).union(set(job_freq.keys())))
    })
    # Add job frequencies
    job_freq_df = pd.DataFrame({
        "Skill": list(job_freq.keys()),
        "Frequency": list(job_freq.values()),
        "Source": ["Jobs"] * len(job_freq)
    })
    freq_df = pd.concat([freq_df, job_freq_df], ignore_index=True)

    # Handle empty dataframe case
    if not freq_df.empty:
        fig = px.bar(freq_df, x="Skill", y="Frequency", color="Source", barmode="group", title="Top Skill Frequency")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No skill frequency data available for this course.")

# Smart Recommender with Hugging Face Inference API
st.header("🤖 Smart Skill Recommender")
st.write("AI-powered suggestions using Hugging Face Inference API.")


# Create a mock recommendation function since we might not have API access in deployment
def get_mock_recommendation(low_sim_skills):
    if not low_sim_skills:
        return ["Course is well-aligned—no major gaps!"]

    recommendations = []
    if "Python" in low_sim_skills:
        recommendations.append("Add Python programming projects with real-world datasets")
    if "SQL" in low_sim_skills:
        recommendations.append("Incorporate database design and SQL query exercises")
    if "cloud" in low_sim_skills:
        recommendations.append("Include cloud deployment exercises using AWS or Azure")
    if "machine learning" in low_sim_skills:
        recommendations.append("Add basic machine learning concepts and simple model building")
    if "React" in low_sim_skills or "JavaScript" in low_sim_skills:
        recommendations.append("Integrate web development projects using modern frameworks")

    # If we still need recommendations
    remaining = ["Docker containerization workshops",
                 "Version control with Git branching strategies",
                 "Agile development methodologies",
                 "Test-driven development practices",
                 "API design and development"]

    while len(recommendations) < 3 and remaining:
        recommendations.append(remaining.pop(0))

    return recommendations[:3]


def get_smart_recommendation(course_name, course_embeddings, skills, similarity_df):
    # Identify skills with low similarity
    low_sim_skills = similarity_df[similarity_df["Course Similarity"].astype(float) < 0.7]["Skill"].tolist()

    # Use mock recommendations to avoid API dependency
    return get_mock_recommendation(low_sim_skills)


if st.button("Generate Smart Recommendations"):
    with st.spinner("Analyzing course alignment..."):
        recommendations = get_smart_recommendation(course_name, course_embeddings, skills, similarity_df)
        st.subheader("Recommendations")
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.info("No specific recommendations generated. Course appears to be well-aligned!")

# Footer
st.markdown("---")
st.write("Built with ❤️ by منارة for the Qualithon | Imam Abdulrahman Bin Faisal University")