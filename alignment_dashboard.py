#alignment_dashboard.py
import streamlit as st
import os
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer, util
import numpy as np
import plotly.express as px
from collections import Counter
from huggingface_hub.inference_api import InferenceApi
import random  # Added for mock recommendations

# Load models
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, fast model for embeddings

# Load datasets
courses_df = pd.read_csv("data/cs_course_outcomes.csv")
jobs_df = pd.read_csv("data/cs_job_postings_500.csv")
standards_files = ["data/cs2023_standards.csv", "data/csta_standards.csv",
                   "data/abet_standards.csv", "data/global_cs_standards.csv"]
standards_df = pd.concat([pd.read_csv(file) for file in standards_files], ignore_index=True)

# Define skill list
skills = ["Python", "Java", "C++", "SQL", "AWS", "Kubernetes", "Docker", "TensorFlow", "React", "Node.js",
          "Git", "Linux", "Spark", "Tableau", "Wireshark", "Solidity", "ROS", "OpenCV", "JavaScript", "HTML",
          "CSS", "MongoDB", "PostgreSQL", "Agile", "Scrum", "encryption", "machine learning", "NLP", "cloud"]

# Extract embeddings
courses_df["Embeddings"] = courses_df["Outcome"].apply(lambda x: model.encode(x))
jobs_df["Embeddings"] = jobs_df["Description"].apply(lambda x: model.encode(x))
standards_df["Embeddings"] = standards_df["Competency"].apply(lambda x: model.encode(x))

# Custom CSS for IAU theme with applied suggestions
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1A2A44; /* Dark blue from IAU header */
        color: #F4E3B2; /* Goldish-white text */
    }
    .stHeader {
        background-color: #1A2A44;
        padding: 10px;
        text-align: center;
        color: #F4E3B2;
    }
    .stSidebar {
        background-color: #1A2A44;
        color: #F4E3B2;
        width: 200px !important;
    }
    .stSidebar .stSelectbox, .stSidebar .stTextArea {
        background-color: #2E4066; /* Darker background for sidebar inputs */
        color: #F4E3B2;
        border: 1px solid #F4E3B2;
    }
    .stSidebar .stSelectbox > div > div, .stSidebar .stTextArea > div > div {
        color: #F4E3B2 !important;
    }
    .stButton>button {
        background-color: #F4E3B2;
        color: #1A2A44;
    }
    .stMetric>span {
        color: #F4E3B2;
        opacity: 1.0 !important;
        text-shadow: 1px 1px 3px #000000, 0 0 5px #FFFFFF; /* Stronger shadow with white outline */
        background-color: rgba(30, 50, 80, 0.9); /* Darker blue background */
        padding: 8px 12px;
        border: 2px solid #F4E3B2; /* Gold border */
        border-radius: 8px;
    }
    .stDataFrame {
        background-color: #2E4066;
        color: #F4E3B2;
        border: 2px solid #F4E3B2;
        border-radius: 5px;
    }
    .stDataFrame th, .stDataFrame td {
        font-weight: bold;
        padding: 8px;
    }
    h1, h2, h3 {
        color: #F4E3B2 !important;
    }
    .css-1aumxhk { /* Target title element */
        color: #F4E3B2 !important;
    }
    /* Remove pink brain icon (replace with neutral/gold if desired) */
    .stApp [data-testid="stDecoration"] {
        display: none;
    }
    /* Add subtle IAU branding to footer */
    .stFooter {
        color: #F4E3B2;
        text-align: center;
        font-size: 12px;
    }
    .stFooter::after {
        content: " | IAU - Jubail";
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add IAU header
st.markdown(
    f"""
    <div class="stHeader">
        <img src="Header_img.png" alt="IAU Header" style="width:100%; height:auto;">
    </div>
    """,
    unsafe_allow_html=True
)

# Streamlit app
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

# KPIs
st.header("📊 Key Performance Indicators")
col1, col2, col3 = st.columns(3)

avg_job_similarity = np.mean(job_similarities) * 100
avg_standard_similarity = np.mean(standard_similarities) * 100
skill_gaps = len([s for s in skills if max([util.cos_sim(model.encode(s), e)[0][0] for e in course_embeddings]) < 0.7])

with col1:
    st.metric("Market Needs Alignment", f"{avg_job_similarity:.2f}%")
with col2:
    st.metric("Standards Alignment", f"{avg_standard_similarity:.2f}%")
with col3:
    st.metric("Semantic Skill Gaps", skill_gaps)

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

# Visualizations
st.header("📈 Visual Insights")
tab1, tab2 = st.tabs(["Similarity Distribution", "Top Skills"])

with tab1:
    sim_df = pd.DataFrame({
        "Similarity": np.concatenate([job_similarities, standard_similarities]),
        "Source": ["Jobs"] * len(job_similarities) + ["Standards"] * len(standard_similarities)
    })
    fig = px.histogram(sim_df, x="Similarity", color="Source", nbins=20, title="Similarity Distribution")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    course_skills = sum(courses_df[courses_df["Course_Name"] == course_name]["Outcome"].apply(
        lambda x: [t.text for t in nlp(x) if t.text in skills]).tolist(), [])
    job_skills = sum(jobs_df["Description"].apply(lambda x: [t.text for t in nlp(x) if t.text in skills]).tolist(), [])
    course_freq = Counter(course_skills)
    job_freq = Counter(job_skills)
    freq_df = pd.DataFrame({
        "Skill": list(course_freq.keys()) + list(job_freq.keys()),
        "Frequency": list(course_freq.values()) + list(job_freq.values()),
        "Source": ["Courses"] * len(course_freq) + ["Jobs"] * len(job_freq)
    })
    fig = px.bar(freq_df, x="Skill", y="Frequency", color="Source", barmode="group", title="Top Skill Frequency")
    st.plotly_chart(fig, use_container_width=True)

# Smart Recommender with Hugging Face Inference API
st.header("🤖 Smart Skill Recommender")
st.write("AI-powered suggestions using Hugging Face Inference API.")

# Initialize Hugging Face Inference API
inference = InferenceApi("google/flan-t5-base", token=os.getenv("HF_API_TOKEN"))  # Optional token for higher limits


def get_smart_recommendation(course_name, course_embeddings, skills, similarity_df):
    low_sim_skills = similarity_df[similarity_df["Course Similarity"].astype(float) < 0.7]["Skill"].tolist()
    if not low_sim_skills:
        return ["Course is well-aligned—no major gaps!"]

    # Prepare prompt for LLM
    course_skills_str = ", ".join(
        [s for s in skills if max([util.cos_sim(model.encode(s), e)[0][0] for e in course_embeddings]) >= 0.7])
    gaps_str = ", ".join(low_sim_skills)
    prompt = f"""
    I have a computer science course named '{course_name}' that teaches these skills: {course_skills_str}.
    It is missing or has low similarity with these skills required by jobs and standards: {gaps_str}.
    Suggest 3 specific, practical enhancements (e.g., new topics, tools, or projects) to improve alignment with industry needs and international standards.
    Keep responses concise, actionable, and relevant to the course's existing skills.
    Output: 1. Enhancement 1 2. Enhancement 2 3. Enhancement 3
    """

    # Call Hugging Face Inference API with raw response
    try:
        response = inference(inputs=prompt, raw_response=True)
        # Parse the raw response text
        recommendations_text = response.text if hasattr(response, 'text') else str(response)
        # Split and clean the response
        if "Output:" in recommendations_text:
            parts = recommendations_text.split("Output:")[1].strip().split()
            recommendations = []
            current_rec = []
            for part in parts:
                if part.isdigit() and len(current_rec) > 0:
                    recommendations.append(" ".join(current_rec).strip())
                    current_rec = []
                current_rec.append(part)
            if current_rec:
                recommendations.append(" ".join(current_rec).strip())
            recommendations = [r for r in recommendations if r and not r.isdigit()]
            if len(recommendations) < 3:
                return [f"Mock suggestion: Add {random.choice(low_sim_skills)} with a project." for _ in
                        range(3 - len(recommendations))] + recommendations[:3]
            return recommendations[:3]
        else:
            return [f"Mock suggestion: Add {random.choice(low_sim_skills)} with a project." for _ in range(3)]
    except Exception as e:
        st.error(f"Error calling Hugging Face API: {e}")
        return [f"Mock suggestion: Add {random.choice(low_sim_skills)} with a project." for _ in range(3)]


if st.button("Generate Smart Recommendations"):
    with st.spinner("Consulting AI..."):
        recommendations = get_smart_recommendation(course_name, course_embeddings, skills, similarity_df)
        st.subheader("Recommendations")
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.error("No recommendations generated. Check API or network.")

# Footer
st.markdown("---")
st.write("Built with ❤️ by منارة for the Qualithon | Imam Abdulrahman Bin Faisal University")