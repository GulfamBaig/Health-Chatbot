import streamlit as st
import pandas as pd
import pickle
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
        return None, None

@st.cache_data
def load_data():
    try:
        return pd.read_csv('medical_qa.csv')
    except Exception as e:
        st.error(f"Error loading data: {e}")
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

    if vectorizer is None or model is None or qa_data.empty:
        st.error("Failed to load required resources. Please check the error messages above.")
        return

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
            return qa_data.loc[qa_data['question'] == pred_class, 'answer'].values[0]
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return "I'm sorry, I couldn't process your question. Please try again."

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
        for question in qa_data['question'].sample(5):
            st.markdown(f"- {question}")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
            This chatbot provides general health information.  
            **Not a substitute** for professional medical advice.
            Always consult a healthcare provider for medical concerns.
        """)

if __name__ == "__main__":
    main()
