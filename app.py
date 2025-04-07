import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the models and data
@st.cache_resource
def load_models():
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('chatbot_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model

@st.cache_data
def load_data():
    return pd.read_csv('medical_qa.csv')

vectorizer, model = load_models()
qa_data = load_data()

# App title and description
st.set_page_config(
    page_title="Health Chatbot",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
        }
        .stTextInput>div>div>input {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 12px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            padding: 10px 24px;
            font-weight: bold;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .chat-message {
            padding: 12px 16px;
            border-radius: 10px;
            margin-bottom: 8px;
            max-width: 80%;
        }
        .user-message {
            background-color: #e3f2fd;
            margin-left: auto;
            border-bottom-right-radius: 4px;
        }
        .bot-message {
            background-color: #ffffff;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            border-bottom-left-radius: 4px;
        }
        .title-text {
            color: #2c3e50;
            font-size: 2.5rem;
            font-weight: 700;
        }
        .subheader-text {
            color: #7f8c8d;
            font-size: 1.1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Header section
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2785/2785482.png", width=80)
with col2:
    st.markdown('<p class="title-text">Health Chatbot</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader-text">Get answers to your medical questions</p>', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to get bot response
def get_response(user_input):
    # Vectorize the input
    input_vec = vectorizer.transform([user_input])
    
    # Predict the most similar question
    pred_class = model.predict(input_vec)[0]
    
    # Get the corresponding answer
    answer = qa_data.loc[qa_data['question'] == pred_class, 'answer'].values[0]
    
    return answer

# Chat input
if prompt := st.chat_input("Ask a health-related question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        response = get_response(prompt)
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with sample questions
with st.sidebar:
    st.markdown("### 💡 Sample Questions")
    st.markdown("Try asking about:")
    for question in qa_data['question'].sample(5):
        st.markdown(f"- {question}")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("This chatbot can answer questions about common health conditions and their symptoms, treatments, and management.")
    st.markdown("**Note:** This is for informational purposes only and not a substitute for professional medical advice.")
