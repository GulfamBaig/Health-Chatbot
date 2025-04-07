import streamlit as st
import pandas as pd
import pickle
import sys
from importlib import import_module

# Check for required packages
def check_dependencies():
    required = {
        'scikit-learn': 'sklearn',
        'pandas': 'pandas',
        'pickle-mixin': 'pickle'
    }
    
    missing = []
    for pkg, module in required.items():
        try:
            import_module(module)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        st.error(f"Missing required packages: {', '.join(missing)}")
        st.markdown("""
            Please install them using:
            ```
            pip install scikit-learn pandas
            ```
            Or add them to your `requirements.txt` file.
        """)
        st.stop()

check_dependencies()

# Now safely import sklearn components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the models and data
@st.cache_resource
def load_models():
    try:
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('chatbot_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return vectorizer, model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error("Please make sure the model files (vectorizer.pkl and chatbot_model.pkl) are in the correct directory.")
        return None, None

@st.cache_data
def load_data():
    try:
        return pd.read_csv('medical_qa.csv')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.error("Please make sure medical_qa.csv exists in the correct directory.")
        return pd.DataFrame(columns=['question', 'answer'])

# Initialize the app
def main():
    st.set_page_config(
        page_title="Health Chatbot",
        page_icon="🩺",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
            .stApp { background-color: #f0f2f6; }
            .stTextInput>div>div>input {
                background-color: #ffffff;
                border-radius: 10px;
                padding: 12px;
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

    # Header
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2785/2785482.png", width=80)
    with col2:
        st.markdown('<p class="title-text">Health Chatbot</p>', unsafe_allow_html=True)
        st.markdown('<p class="subheader-text">Get answers to your medical questions</p>', unsafe_allow_html=True)

    # Load resources
    vectorizer, model = load_models()
    qa_data = load_data()

    if vectorizer is None or model is None:
        st.error("Failed to load AI models. The chatbot cannot function without these.")
        st.stop()

    if qa_data.empty:
        st.error("Failed to load medical Q&A data. The chatbot cannot function without this.")
        st.stop()

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    def get_response(user_input):
        try:
            input_vec = vectorizer.transform([user_input])
            pred_class = model.predict(input_vec)[0]
            answers = qa_data.loc[qa_data['question'] == pred_class, 'answer']
            if not answers.empty:
                return answers.values[0]
            return "I couldn't find a specific answer to that question. Please try rephrasing."
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return "I'm having trouble answering that. Please try a different question."

    if prompt := st.chat_input("Ask a health-related question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            response = get_response(prompt)
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Sidebar
    with st.sidebar:
        st.markdown("### 💡 Sample Questions")
        st.markdown("Try asking about:")
        for question in qa_data['question'].sample(min(5, len(qa_data))):  # Handle case with <5 questions
            st.markdown(f"- {question}")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
            This chatbot provides general health information.  
            **Not a substitute** for professional medical advice.
            Always consult a healthcare provider for medical concerns.
        """)
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()
