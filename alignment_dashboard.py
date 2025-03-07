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
# Header image
st.image("Header_img.png", use_column_width=True)

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