import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import random
import json
import time

# Set page configuration and title
st.set_page_config(
    page_title="Smart CS Alignment Dashboard",
    page_icon="🧠",
    layout="wide"
)


# Simulate generated data for demonstration purposes
def create_mock_courses():
    return pd.DataFrame({
        "Course_Name": ["Introduction to Programming", "Data Structures", "Algorithms", "Web Development",
                        "Database Systems"],
        "Outcome": [
            "Students will learn fundamentals of programming using Python, including variables, loops, conditionals, and basic data structures. Students will develop simple command-line applications.",
            "Students will implement and analyze various data structures including arrays, linked lists, trees, and graphs. Focus is on abstract data types and algorithm complexity analysis.",
            "Students will design and analyze algorithms for problem-solving, focusing on computational complexity. Topics include sorting, searching, and graph algorithms.",
            "Students will learn HTML, CSS, and JavaScript to build responsive websites. Focus is on front-end development and basic design principles.",
            "Students will design and implement relational database systems using SQL. Topics include normalization, query optimization, and transaction management."
        ]
    })


def create_mock_jobs():
    return pd.DataFrame({
        "Title": ["Software Engineer", "Data Scientist", "Web Developer", "DevOps Engineer", "Database Administrator"],
        "Description": [
            "Develop software using Python, Java, and SQL. Knowledge of cloud platforms (AWS, Azure), CI/CD pipelines, and Git version control required. Experience with Agile methodologies and containerization (Docker, Kubernetes) preferred.",
            "Analyze data using Python, R, and machine learning frameworks (TensorFlow, PyTorch). Must have strong data visualization skills (Tableau, PowerBI). Experience with big data tools (Spark, Hadoop) and cloud-based data solutions preferred.",
            "Create web applications using JavaScript, React, Node.js, HTML5, and CSS3. Experience with responsive design, SEO, and web performance optimization. Knowledge of modern frameworks, REST APIs, and GraphQL required.",
            "Manage infrastructure using AWS, Kubernetes, Docker, and Terraform. Experience with automation, infrastructure as code, monitoring tools, and security best practices. Strong Linux skills and scripting abilities required.",
            "Manage and optimize PostgreSQL and MySQL database systems. Experience with performance tuning, data modeling, backup solutions, and high availability configurations. Knowledge of NoSQL databases (MongoDB, Cassandra) a plus."
        ]
    })


def create_mock_standards():
    return pd.DataFrame({
        "Standard": ["CS2023-1", "CSTA-2", "ABET-3", "CSEC-4", "ACM-5"],
        "Competency": [
            "Apply programming fundamentals to solve problems using procedural and object-oriented paradigms. Must include testing, debugging, and quality assurance practices.",
            "Design and implement data structures and algorithms to optimize computational efficiency. Should include analysis of algorithms and performance considerations.",
            "Demonstrate team collaboration, project management, and professional ethics in software development projects. Must include version control systems and CI/CD practices.",
            "Apply security principles and practices in software development and system design. Should cover encryption, authentication, and secure coding standards.",
            "Develop applications utilizing cloud services, containerization, and microservice architectures. Should include deployment strategies and scalability considerations."
        ]
    })


# Load datasets
courses_df = create_mock_courses()
jobs_df = create_mock_jobs()
standards_df = create_mock_standards()

# Define skill categories
skills_by_category = {
    "Programming Languages": ["Python", "Java", "C++", "JavaScript", "R", "TypeScript"],
    "Web Technologies": ["HTML", "CSS", "React", "Angular", "Node.js", "REST API"],
    "Databases": ["SQL", "MongoDB", "PostgreSQL", "MySQL", "NoSQL"],
    "Cloud & DevOps": ["AWS", "Docker", "Kubernetes", "Git", "CI/CD", "Linux"],
    "Data Science & AI": ["TensorFlow", "PyTorch", "machine learning", "NLP", "data visualization"],
    "Tools & Practices": ["Agile", "Scrum", "testing", "debugging", "version control", "security"]
}

# Flatten skills list while preserving category information
skills_with_categories = [(skill, category) for category, skill_list in skills_by_category.items() for skill in
                          skill_list]
skills = [skill for skill, _ in skills_with_categories]
skill_to_category = {skill: category for skill, category in skills_with_categories}


# Function to extract skills using regex (simplified for demo)
def extract_skills(text, skill_list):
    text = text.lower()
    found_skills = []
    for skill in skill_list:
        # Create a regex pattern that matches the skill as a whole word
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text):
            found_skills.append(skill)
    return found_skills


# Precomputed similarity scores for demonstration (simulating embeddings)
# These would normally come from a model like SentenceTransformer
def get_simulated_scores():
    # Structure: {course_name: {skill: {course_score, job_score, standard_score}}}
    return {
        "Introduction to Programming": {
            "Python": {"course": 0.85, "job": 0.90, "standard": 0.75},
            "Java": {"course": 0.30, "job": 0.85, "standard": 0.65},
            "JavaScript": {"course": 0.20, "job": 0.80, "standard": 0.60},
            "C++": {"course": 0.25, "job": 0.75, "standard": 0.70},
            "R": {"course": 0.15, "job": 0.65, "standard": 0.40},
            "TypeScript": {"course": 0.10, "job": 0.70, "standard": 0.45},
            "HTML": {"course": 0.25, "job": 0.75, "standard": 0.50},
            "CSS": {"course": 0.20, "job": 0.75, "standard": 0.50},
            "React": {"course": 0.10, "job": 0.80, "standard": 0.55},
            "Angular": {"course": 0.05, "job": 0.70, "standard": 0.45},
            "Node.js": {"course": 0.10, "job": 0.75, "standard": 0.50},
            "REST API": {"course": 0.15, "job": 0.80, "standard": 0.60},
            "SQL": {"course": 0.30, "job": 0.85, "standard": 0.70},
            "MongoDB": {"course": 0.10, "job": 0.75, "standard": 0.50},
            "PostgreSQL": {"course": 0.05, "job": 0.70, "standard": 0.45},
            "MySQL": {"course": 0.15, "job": 0.70, "standard": 0.50},
            "NoSQL": {"course": 0.05, "job": 0.65, "standard": 0.45},
            "AWS": {"course": 0.15, "job": 0.80, "standard": 0.55},
            "Docker": {"course": 0.10, "job": 0.75, "standard": 0.60},
            "Kubernetes": {"course": 0.05, "job": 0.70, "standard": 0.50},
            "Git": {"course": 0.30, "job": 0.85, "standard": 0.75},
            "CI/CD": {"course": 0.10, "job": 0.75, "standard": 0.65},
            "Linux": {"course": 0.25, "job": 0.80, "standard": 0.60},
            "TensorFlow": {"course": 0.05, "job": 0.70, "standard": 0.45},
            "PyTorch": {"course": 0.05, "job": 0.65, "standard": 0.40},
            "machine learning": {"course": 0.15, "job": 0.75, "standard": 0.55},
            "NLP": {"course": 0.05, "job": 0.65, "standard": 0.40},
            "data visualization": {"course": 0.25, "job": 0.70, "standard": 0.55},
            "Agile": {"course": 0.15, "job": 0.80, "standard": 0.70},
            "Scrum": {"course": 0.10, "job": 0.75, "standard": 0.65},
            "testing": {"course": 0.35, "job": 0.85, "standard": 0.75},
            "debugging": {"course": 0.70, "job": 0.80, "standard": 0.75},
            "version control": {"course": 0.30, "job": 0.85, "standard": 0.70},
            "security": {"course": 0.20, "job": 0.80, "standard": 0.75}
        },
        "Data Structures": {
            "Python": {"course": 0.70, "job": 0.90, "standard": 0.75},
            "Java": {"course": 0.85, "job": 0.85, "standard": 0.65},
            "JavaScript": {"course": 0.15, "job": 0.80, "standard": 0.60},
            "C++": {"course": 0.75, "job": 0.75, "standard": 0.70},
            "R": {"course": 0.10, "job": 0.65, "standard": 0.40},
            "TypeScript": {"course": 0.05, "job": 0.70, "standard": 0.45},
            "HTML": {"course": 0.05, "job": 0.75, "standard": 0.50},
            "CSS": {"course": 0.05, "job": 0.75, "standard": 0.50},
            "React": {"course": 0.05, "job": 0.80, "standard": 0.55},
            "Angular": {"course": 0.05, "job": 0.70, "standard": 0.45},
            "Node.js": {"course": 0.05, "job": 0.75, "standard": 0.50},
            "REST API": {"course": 0.10, "job": 0.80, "standard": 0.60},
            "SQL": {"course": 0.15, "job": 0.85, "standard": 0.70},
            "MongoDB": {"course": 0.05, "job": 0.75, "standard": 0.50},
            "PostgreSQL": {"course": 0.05, "job": 0.70, "standard": 0.45},
            "MySQL": {"course": 0.10, "job": 0.70, "standard": 0.50},
            "NoSQL": {"course": 0.05, "job": 0.65, "standard": 0.45},
            "AWS": {"course": 0.05, "job": 0.80, "standard": 0.55},
            "Docker": {"course": 0.05, "job": 0.75, "standard": 0.60},
            "Kubernetes": {"course": 0.05, "job": 0.70, "standard": 0.50},
            "Git": {"course": 0.20, "job": 0.85, "standard": 0.75},
            "CI/CD": {"course": 0.05, "job": 0.75, "standard": 0.65},
            "Linux": {"course": 0.15, "job": 0.80, "standard": 0.60},
            "TensorFlow": {"course": 0.05, "job": 0.70, "standard": 0.45},
            "PyTorch": {"course": 0.05, "job": 0.65, "standard": 0.40},
            "machine learning": {"course": 0.10, "job": 0.75, "standard": 0.55},
            "NLP": {"course": 0.05, "job": 0.65, "standard": 0.40},
            "data visualization": {"course": 0.10, "job": 0.70, "standard": 0.55},
            "Agile": {"course": 0.10, "job": 0.80, "standard": 0.70},
            "Scrum": {"course": 0.05, "job": 0.75, "standard": 0.65},
            "testing": {"course": 0.30, "job": 0.85, "standard": 0.75},
            "debugging": {"course": 0.40, "job": 0.80, "standard": 0.75},
            "version control": {"course": 0.15, "job": 0.85, "standard": 0.70},
            "security": {"course": 0.10, "job": 0.80, "standard": 0.75}
        },
        "Algorithms": {
            "Python": {"course": 0.60, "job": 0.90, "standard": 0.75},
            "Java": {"course": 0.65, "job": 0.85, "standard": 0.65},
            "JavaScript": {"course": 0.10, "job": 0.80, "standard": 0.60},
            "C++": {"course": 0.70, "job": 0.75, "standard": 0.70},
            "R": {"course": 0.15, "job": 0.65, "standard": 0.40},
            "TypeScript": {"course": 0.05, "job": 0.70, "standard": 0.45},
            "HTML": {"course": 0.05, "job": 0.75, "standard": 0.50},
            "CSS": {"course": 0.05, "job": 0.75, "standard": 0.50},
            "React": {"course": 0.05, "job": 0.80, "standard": 0.55},
            "Angular": {"course": 0.05, "job": 0.70, "standard": 0.45},
            "Node.js": {"course": 0.05, "job": 0.75, "standard": 0.50},
            "REST API": {"course": 0.10, "job": 0.80, "standard": 0.60},
            "SQL": {"course": 0.15, "job": 0.85, "standard": 0.70},
            "MongoDB": {"course": 0.05, "job": 0.75, "standard": 0.50},
            "PostgreSQL": {"course": 0.05, "job": 0.70, "standard": 0.45},
            "MySQL": {"course": 0.05, "job": 0.70, "standard": 0.50},
            "NoSQL": {"course": 0.05, "job": 0.65, "standard": 0.45},
            "AWS": {"course": 0.05, "job": 0.80, "standard": 0.55},
            "Docker": {"course": 0.05, "job": 0.75, "standard": 0.60},
            "Kubernetes": {"course": 0.05, "job": 0.70, "standard": 0.50},
            "Git": {"course": 0.15, "job": 0.85, "standard": 0.75},
            "CI/CD": {"course": 0.05, "job": 0.75, "standard": 0.65},
            "Linux": {"course": 0.10, "job": 0.80, "standard": 0.60},
            "TensorFlow": {"course": 0.20, "job": 0.70, "standard": 0.45},
            "PyTorch": {"course": 0.15, "job": 0.65, "standard": 0.40},
            "machine learning": {"course": 0.35, "job": 0.75, "standard": 0.55},
            "NLP": {"course": 0.25, "job": 0.65, "standard": 0.40},
            "data visualization": {"course": 0.10, "job": 0.70, "standard": 0.55},
            "Agile": {"course": 0.05, "job": 0.80, "standard": 0.70},
            "Scrum": {"course": 0.05, "job": 0.75, "standard": 0.65},
            "testing": {"course": 0.30, "job": 0.85, "standard": 0.75},
            "debugging": {"course": 0.35, "job": 0.80, "standard": 0.75},
            "version control": {"course": 0.10, "job": 0.85, "standard": 0.70},
            "security": {"course": 0.10, "job": 0.80, "standard": 0.75}
        },
        "Web Development": {
            "Python": {"course": 0.35, "job": 0.90, "standard": 0.75},
            "Java": {"course": 0.25, "job": 0.85, "standard": 0.65},
            "JavaScript": {"course": 0.90, "job": 0.80, "standard": 0.60},
            "C++": {"course": 0.10, "job": 0.75, "standard": 0.70},
            "R": {"course": 0.05, "job": 0.65, "standard": 0.40},
            "TypeScript": {"course": 0.60, "job": 0.70, "standard": 0.45},
            "HTML": {"course": 0.95, "job": 0.75, "standard": 0.50},
            "CSS": {"course": 0.95, "job": 0.75, "standard": 0.50},
            "React": {"course": 0.75, "job": 0.80, "standard": 0.55},
            "Angular": {"course": 0.50, "job": 0.70, "standard": 0.45},
            "Node.js": {"course": 0.70, "job": 0.75, "standard": 0.50},
            "REST API": {"course": 0.65, "job": 0.80, "standard": 0.60},
            "SQL": {"course": 0.30, "job": 0.85, "standard": 0.70},
            "MongoDB": {"course": 0.55, "job": 0.75, "standard": 0.50},
            "PostgreSQL": {"course": 0.25, "job": 0.70, "standard": 0.45},
            "MySQL": {"course": 0.40, "job": 0.70, "standard": 0.50},
            "NoSQL": {"course": 0.40, "job": 0.65, "standard": 0.45},
            "AWS": {"course": 0.25, "job": 0.80, "standard": 0.55},
            "Docker": {"course": 0.20, "job": 0.75, "standard": 0.60},
            "Kubernetes": {"course": 0.05, "job": 0.70, "standard": 0.50},
            "Git": {"course": 0.60, "job": 0.85, "standard": 0.75},
            "CI/CD": {"course": 0.25, "job": 0.75, "standard": 0.65},
            "Linux": {"course": 0.20, "job": 0.80, "standard": 0.60},
            "TensorFlow": {"course": 0.05, "job": 0.70, "standard": 0.45},
            "PyTorch": {"course": 0.05, "job": 0.65, "standard": 0.40},
            "machine learning": {"course": 0.10, "job": 0.75, "standard": 0.55},
            "NLP": {"course": 0.05, "job": 0.65, "standard": 0.40},
            "data visualization": {"course": 0.40, "job": 0.70, "standard": 0.55},
            "Agile": {"course": 0.35, "job": 0.80, "standard": 0.70},
            "Scrum": {"course": 0.30, "job": 0.75, "standard": 0.65},
            "testing": {"course": 0.40, "job": 0.85, "standard": 0.75},
            "debugging": {"course": 0.55, "job": 0.80, "standard": 0.75},
            "version control": {"course": 0.55, "job": 0.85, "standard": 0.70},
            "security": {"course": 0.30, "job": 0.80, "standard": 0.75}
        },
        "Database Systems": {
            "Python": {"course": 0.40, "job": 0.90, "standard": 0.75},
            "Java": {"course": 0.25, "job": 0.85, "standard": 0.65},
            "JavaScript": {"course": 0.15, "job": 0.80, "standard": 0.60},
            "C++": {"course": 0.15, "job": 0.75, "standard": 0.70},
            "R": {"course": 0.20, "job": 0.65, "standard": 0.40},
            "TypeScript": {"course": 0.05, "job": 0.70, "standard": 0.45},
            "HTML": {"course": 0.10, "job": 0.75, "standard": 0.50},
            "CSS": {"course": 0.05, "job": 0.75, "standard": 0.50},
            "React": {"course": 0.05, "job": 0.80, "standard": 0.55},
            "Angular": {"course": 0.05, "job": 0.70, "standard": 0.45},
            "Node.js": {"course": 0.15, "job": 0.75, "standard": 0.50},
            "REST API": {"course": 0.30, "job": 0.80, "standard": 0.60},
            "SQL": {"course": 0.95, "job": 0.85, "standard": 0.70},
            "MongoDB": {"course": 0.55, "job": 0.75, "standard": 0.50},
            "PostgreSQL": {"course": 0.75, "job": 0.70, "standard": 0.45},
            "MySQL": {"course": 0.90, "job": 0.70, "standard": 0.50},
            "NoSQL": {"course": 0.65, "job": 0.65, "standard": 0.45},
            "AWS": {"course": 0.25, "job": 0.80, "standard": 0.55},
            "Docker": {"course": 0.15, "job": 0.75, "standard": 0.60},
            "Kubernetes": {"course": 0.05, "job": 0.70, "standard": 0.50},
            "Git": {"course": 0.20, "job": 0.85, "standard": 0.75},
            "CI/CD": {"course": 0.10, "job": 0.75, "standard": 0.65},
            "Linux": {"course": 0.30, "job": 0.80, "standard": 0.60},
            "TensorFlow": {"course": 0.10, "job": 0.70, "standard": 0.45},
            "PyTorch": {"course": 0.05, "job": 0.65, "standard": 0.40},
            "machine learning": {"course": 0.25, "job": 0.75, "standard": 0.55},
            "NLP": {"course": 0.10, "job": 0.65, "standard": 0.40},
            "data visualization": {"course": 0.45, "job": 0.70, "standard": 0.55},
            "Agile": {"course": 0.15, "job": 0.80, "standard": 0.70},
            "Scrum": {"course": 0.10, "job": 0.75, "standard": 0.65},
            "testing": {"course": 0.40, "job": 0.85, "standard": 0.75},
            "debugging": {"course": 0.50, "job": 0.80, "standard": 0.75},
            "version control": {"course": 0.20, "job": 0.85, "standard": 0.70},
            "security": {"course": 0.55, "job": 0.80, "standard": 0.75}
        }
    }


# Simulated LLM API call for recommendations
def simulate_llm_api_call(course_name, gaps, thresholds):
    # In a real implementation, this would be an API call to an LLM service
    # For demo purposes, we'll simulate the API response

    st.spinner("Calling AI Recommendation Engine...")

    # Simulate API call delay
    time.sleep(2)

    # Pre-written course-specific recommendations
    course_recommendations = {
        "Introduction to Programming": [
            f"Integrate Git version control into programming assignments to teach students industry-standard collaboration practices. Setup a classroom GitHub organization with repositories for each project.",
            f"Add a module on REST API fundamentals using Python requests library to introduce students to web services concepts. Implement simple API projects using public APIs.",
            f"Introduce Java as a secondary language with comparisons to Python to broaden students' programming language knowledge. Create parallel assignments to implement the same solution in both languages."
        ],
        "Data Structures": [
            f"Add practical applications of data structures in web development contexts using JavaScript, showing how structures like trees and graphs apply to DOM manipulation and state management.",
            f"Incorporate version control using Git throughout coursework, requiring proper commit messages and branching strategies for collaborative data structure implementations.",
            f"Introduce cloud-based implementations (AWS) of data structures to demonstrate scalability considerations with large datasets that exceed local memory constraints."
        ],
        "Algorithms": [
            f"Implement a module on machine learning algorithms that builds on the theoretical foundations already covered in the course. Show connections between classic algorithms and their ML applications.",
            f"Add a practical project applying algorithms to security challenges, including encryption, authentication, and secure coding practices to address the gap in security knowledge.",
            f"Incorporate visualization exercises using data visualization tools to help students understand algorithm performance and complexity in real-world scenarios."
        ],
        "Web Development": [
            f"Enhance the curriculum with Docker containerization for web applications, teaching students to create reproducible development environments and deployment packages.",
            f"Add a module on serverless deployment using AWS Lambda and API Gateway to teach modern cloud-based web application architecture and scaling.",
            f"Incorporate security-focused exercises addressing common web vulnerabilities (XSS, CSRF, SQL injection) to improve students' defensive coding practices."
        ],
        "Database Systems": [
            f"Integrate cloud database solutions (AWS RDS, DynamoDB) alongside traditional database systems to teach modern deployment and scaling considerations.",
            f"Add a CI/CD pipeline component for database schema migrations and automated testing to expose students to DevOps practices specific to database management.",
            f"Incorporate a module on data visualization and dashboard creation to connect database knowledge with practical reporting and business intelligence applications."
        ]
    }

    # Format recommendations based on selected gaps
    if course_name in course_recommendations:
        return course_recommendations[course_name]
    else:
        # Generic recommendations if course not found
        return [
            "Incorporate version control and collaborative development practices using Git and GitHub Classroom.",
            "Add industry-relevant projects that connect theoretical concepts to practical applications.",
            "Integrate cloud computing concepts and tools to prepare students for modern development environments."
        ]


# Custom CSS for dashboard theme
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
    .metrics-card {
        background-color: rgba(30, 50, 80, 1.0);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #F4E3B2;
        text-align: center;
    }
    .metrics-label {
        color: #FFFF00;
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .metrics-value {
        color: #FFFFFF;
        font-size: 24px;
        font-weight: bold;
    }
    .recommendation-card {
        background-color: rgba(30, 50, 80, 0.6);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #F4E3B2;
        margin-bottom: 10px;
    }
    .recommendation-title {
        font-weight: bold;
        color: #FFFF00;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add header image or logo
st.markdown("""

<div style="display: flex; justify-content: space-between; align-items: center; padding: 10px; background-color: #2E4066; border-radius: 10px; margin-bottom: 20px;">
    <div>
        <h1 style="margin: 0; padding: 0;">🧠 Smart CS Alignment Dashboard</h1>
        <p style="margin: 5px 0 0 0;">Using AI to align curriculum with industry needs and standards</p>
    </div>
    <div>
        <h3 style="margin: 0;">IAU University</h3>
        <p style="margin: 0;">Computer Science Department</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar for course selection
st.sidebar.header("Course Selection")
course_options = courses_df["Course_Name"].unique()
selected_course = st.sidebar.selectbox("Select a Course", course_options)
custom_outcome = st.sidebar.text_area("Or Enter Custom Outcome", "")

# Filters
st.sidebar.header("Analysis Settings")
selected_categories = st.sidebar.multiselect(
    "Skill Categories",
    options=list(skills_by_category.keys()),
    default=list(skills_by_category.keys())
)

# Thresholds with tooltips
st.sidebar.markdown("##### Analysis Thresholds")
st.sidebar.markdown("_Adjust sensitivity of gap detection_")
threshold_job = st.sidebar.slider("Job Market Relevance", 0.0, 1.0, 0.5,
                                  help="Minimum similarity score for a skill to be considered relevant in job market")
threshold_course = st.sidebar.slider("Course Coverage", 0.0, 1.0, 0.3,
                                     help="Minimum similarity score for a skill to be considered covered in the course")
gap_severity = st.sidebar.slider("Gap Severity", 0.0, 1.0, 0.2,
                                 help="Minimum difference between job relevance and course coverage to flag as a gap")

# Simulation of model loading
if "loaded" not in st.session_state:
    with st.spinner("Loading AI models and computing embeddings..."):
        time.sleep(2)  # Simulate loading time
        st.session_state.loaded = True

# Get course text
if custom_outcome:
    course_text = custom_outcome
    course_name = "Custom Course"
else:
    course_data = courses_df[courses_df["Course_Name"] == selected_course]
    course_text = course_data["Outcome"].iloc[0]
    course_name = selected_course

# Get simulated scores for the selected course
simulated_scores = get_simulated_scores()
course_scores = simulated_scores.get(course_name, {})
if course_name == "Custom Course":
    # Use Introduction to Programming as a fallback for custom input
    course_scores = simulated_scores["Introduction to Programming"]

# Filter skills based on selected categories
filtered_skills = []
for category in selected_categories:
    filtered_skills.extend(skills_by_category[category])

# Identify skill gaps
skill_gaps = []
for skill in filtered_skills:
    if skill in course_scores:
        course_score = course_scores[skill]["course"]
        job_score = course_scores[skill]["job"]
        standard_score = course_scores[skill]["standard"]

        # Calculate gap severity
        job_gap = max(0, job_score - course_score)
        standard_gap = max(0, standard_score - course_score)

        # Calculate overall gap (weighted more toward job market)
        overall_gap = (job_gap * 0.7) + (standard_gap * 0.3)

        # Check if this skill has a significant gap
        if job_score >= threshold_job and course_score < threshold_course and overall_gap >= gap_severity:
            skill_gaps.append({
                "skill": skill,
                "category": skill_to_category.get(skill, "Other"),
                "course_score": course_score,
                "job_score": job_score,
                "standard_score": standard_score,
                "overall_gap": overall_gap
            })

# Sort gaps by severity
skill_gaps.sort(key=lambda x: x["overall_gap"], reverse=True)

# Calculate coverage metrics
job_relevant_skills = [s for s in filtered_skills if s in course_scores and course_scores[s]["job"] >= threshold_job]
covered_skills = [s for s in job_relevant_skills if course_scores[s]["course"] >= threshold_course]

market_coverage = len(covered_skills) / len(job_relevant_skills) * 100 if job_relevant_skills else 0
standard_relevant = [s for s in filtered_skills if s in course_scores and course_scores[s]["standard"] >= threshold_job]
standard_covered = [s for s in standard_relevant if course_scores[s]["course"] >= threshold_course]
standards_coverage = len(standard_covered) / len(standard_relevant) * 100 if standard_relevant else 0

# Main content
st.header(f"Analysis of: {course_name}")
st.markdown(f"**Course Description:** {course_text}")

# Key metrics in nice cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metrics-card">
        <div class="metrics-label">Market Alignment</div>
        <div class="metrics-value">{market_coverage:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metrics-card">
        <div class="metrics-label">Standards Alignment</div>
        <div class="metrics-value">{standards_coverage:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metrics-card">
        <div class="metrics-label">Skill Gaps</div>
        <div class="metrics-value">{len(skill_gaps)}</div>
    </div>
    """, unsafe_allow_html=True)

# Progress bar explanation
st.markdown("""
---
### Overall Curriculum Health
""")

health_score = (market_coverage + standards_coverage) / 2
st.progress(health_score / 100)
st.markdown(f"<div style='color: #F4E3B2; font-size: 16px;'>Overall curriculum health score: {health_score:.1f}%</div>", unsafe_allow_html=True)

# Display skill gaps in a sortable table
st.header("🔍 Identified Skill Gaps")

if skill_gaps:
    # Format data for display
    gaps_df = pd.DataFrame(skill_gaps)
    gaps_df["Course Coverage"] = gaps_df["course_score"].apply(lambda x: f"{x * 100:.1f}%")
    gaps_df["Job Relevance"] = gaps_df["job_score"].apply(lambda x: f"{x * 100:.1f}%")
    gaps_df["Standards Relevance"] = gaps_df["standard_score"].apply(lambda x: f"{x * 100:.1f}%")
    gaps_df["Gap Severity"] = gaps_df["overall_gap"].apply(lambda x: f"{x * 100:.1f}%")

    display_df = gaps_df[
        ["skill", "category", "Course Coverage", "Job Relevance", "Standards Relevance", "Gap Severity"]]
    display_df.columns = ["Skill", "Category", "Course Coverage", "Job Relevance", "Standards Relevance",
                          "Gap Severity"]

    st.dataframe(display_df, use_container_width=True)

    # Visualization of gaps
    st.header("📊 Gap Visualization")

    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Skills Gap Analysis", "Category Coverage", "Comparison"])

    with tab1:
        # Get top N gaps for visualization
        top_gaps = skill_gaps[:10] if len(skill_gaps) > 10 else skill_gaps

        # Create a horizontal bar chart for gaps
        fig = go.Figure()

        # Add course coverage bars
        fig.add_trace(go.Bar(
            y=[g["skill"] for g in top_gaps],
            x=[g["course_score"] for g in top_gaps],
            name="Course Coverage",
            orientation='h',
            marker=dict(color='rgba(50, 171, 96, 0.7)'),
        ))

        # Add job relevance bars
        fig.add_trace(go.Bar(
            y=[g["skill"] for g in top_gaps],
            x=[g["job_score"] for g in top_gaps],
            name="Job Relevance",
            orientation='h',
            marker=dict(color='rgba(219, 64, 82, 0.7)'),
        ))

        # Update layout
        fig.update_layout(
            title=dict(
                text="Top Skill Gaps: Course Coverage vs. Job Relevance",
                font=dict(color='#F4E3B2', size=18)
            ),
            barmode='group',
            height=500,
            yaxis=dict(title=""),
            xaxis=dict(title="Score"),
            legend=dict(orientation="h"),
            plot_bgcolor='rgba(30, 50, 80, 0.4)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F4E3B2')
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Prepare data by category
        category_data = {}
        for skill in filtered_skills:
            if skill in course_scores:
                category = skill_to_category.get(skill, "Other")
                if category not in category_data:
                    category_data[category] = {
                        "total": 0,
                        "covered": 0,
                        "job_relevant": 0
                    }

                category_data[category]["total"] += 1

                if course_scores[skill]["job"] >= threshold_job:
                    category_data[category]["job_relevant"] += 1
                    if course_scores[skill]["course"] >= threshold_course:
                        category_data[category]["covered"] += 1

        # Calculate coverage percentages
        categories = []
        coverage_pcts = []
        for category, data in category_data.items():
            if data["job_relevant"] > 0:
                categories.append(category)
                coverage_pcts.append((data["covered"] / data["job_relevant"]) * 100)

        # Create radar chart for category coverage
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=coverage_pcts,
            theta=categories,
            fill='toself',
            name='Coverage %',
            line_color='#F4E3B2',
            fillcolor='rgba(244, 227, 178, 0.3)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title=dict(
                text="Coverage by Skill Category",
                font=dict(color='#F4E3B2', size=18)
            ),
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F4E3B2')
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Extract skills for comparison
        course_skills = extract_skills(course_text.lower(), filtered_skills)

        # Aggregate job skills
        job_skills = []
        for desc in jobs_df["Description"]:
            job_skills.extend(extract_skills(desc.lower(), filtered_skills))

        # Count occurrences
        course_skill_counts = Counter(course_skills)
        job_skill_counts = Counter(job_skills)

        # Prepare data for visualization
        all_skills = set(course_skill_counts.keys()) | set(job_skill_counts.keys())
        comparison_data = []

        for skill in all_skills:
            comparison_data.append({
                "Skill": skill,
                "Course Mentions": course_skill_counts.get(skill, 0),
                "Job Mentions": job_skill_counts.get(skill, 0)
            })

        # Sort by job mentions
        comparison_data.sort(key=lambda x: x["Job Mentions"], reverse=True)
        comparison_data = comparison_data[:15]  # Top 15 skills

        # Create comparative bar chart
        fig = px.bar(
            comparison_data,
            x="Skill",
            y=["Course Mentions", "Job Mentions"],
            # Remove title here
            barmode="group",
            height=500
        )

        fig.update_layout(
            title=dict(
                text="Skill Mentions: Course vs. Job Postings",
                font=dict(color='#F4E3B2', size=20, family="Arial, sans-serif"),
                x=0,  # Center the title
                y=0.95  # Position slightly down from the top
            ),
            plot_bgcolor='rgba(30, 50, 80, 0.4)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F4E3B2', size=14)  # Increase overall font size
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info(
        "No significant skill gaps detected with current threshold settings. Try adjusting the thresholds to identify potential areas for improvement.")

# AI Recommendations section
st.header("🤖 AI-Powered Course Enhancement Recommendations")
st.markdown("Use our LLM-based AI to generate tailored recommendations for addressing identified skill gaps")

if st.button("Generate AI Recommendations"):
    # Show a spinner and simulate API call
    with st.spinner("Calling AI Recommendation Engine..."):
        # Simulate API call delay
        time.sleep(2)

        # Call simulated LLM API
        thresholds = {
            "job_relevance": threshold_job,
            "course_coverage": threshold_course,
            "gap_severity": gap_severity
        }

        recommendations = simulate_llm_api_call(course_name, skill_gaps, thresholds)

        # Display API response in a formatted way
        st.markdown("### AI-Generated Recommendations")

        st.markdown("""
        <div style="background-color: rgba(50, 171, 96, 0.1); padding: 10px; border-radius: 5px; margin-bottom: 20px;">
            <p style="font-style: italic; margin: 0;">Recommendations are based on analysis of current industry needs, academic standards, and identified skill gaps in your course.</p>
        </div>
        """, unsafe_allow_html=True)

        # Display each recommendation
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"""
            <div class="recommendation-card">
                <span class="recommendation-title">Recommendation {i}:</span> {rec}
            </div>
            """, unsafe_allow_html=True)

        # Show simulated API call details for demo
        with st.expander("View API Call Details (Demo Only)"):
            st.code(json.dumps({
                "api_endpoint": "https://api.llmprovider.com/recommendations",
                "request": {
                    "course_name": course_name,
                    "gap_count": len(skill_gaps),
                    "top_gaps": [g["skill"] for g in skill_gaps[:5]] if skill_gaps else [],
                    "thresholds": thresholds
                },
                "response": {
                    "recommendations": recommendations,
                    "model": "gpt-4-turbo",
                    "tokens": {
                        "prompt": 1247,
                        "completion": 865,
                        "total": 2112
                    }
                }
            }, indent=2))

# Add explanatory section for committee
with st.expander("About This Dashboard (For Committee Review)"):
    st.markdown("""
    ## Smart CS Alignment Dashboard

    This dashboard demonstrates core functionality for a curriculum alignment tool that:

    1. **Analyzes course descriptions** against job postings and academic standards
    2. **Identifies skill gaps** based on semantic similarity and customizable thresholds
    3. **Visualizes alignment** across different skill categories
    4. **Generates targeted recommendations** using LLM API integration

    ### Technical Approach

    - **NLP Analysis**: Uses semantic similarity to compare course content with job requirements
    - **Skill Gap Detection**: Identifies skills important in industry but missing from curriculum
    - **Category-Based Analysis**: Groups skills into meaningful categories for better insights
    - **LLM Integration**: Leverages large language models to generate contextual recommendations

    ### Implementation Notes

    In a production implementation, this dashboard would:
    - Process real course descriptions, job postings, and standards documents
    - Use actual embedding models for semantic analysis
    - Make live API calls to LLM services for recommendations
    - Include data refreshing capabilities to keep job market analysis current

    The simulated version demonstrates all core functionalities while using pre-computed values for quick demonstration purposes.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center;">
    <div>Built with ❤️ by منارة for the Qualithon</div>
    <div>Imam Abdulrahman Bin Faisal University</div>
</div>
""", unsafe_allow_html=True)