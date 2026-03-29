import streamlit as st
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Knowledge base
questions = [
    "what is artificial intelligence",
    "what is machine learning",
    "what is natural language processing",
    "what is chatbot",
    "who created python",
    "what is deep learning",
    "what is data science"
]

answers = [
    "Artificial Intelligence enables machines to simulate human intelligence.",
    "Machine Learning allows systems to learn from data.",
    "Natural Language Processing allows computers to understand human language.",
    "A chatbot is a program that simulates conversation with users.",
    "Python was created by Guido van Rossum.",
    "Deep learning is a type of machine learning based on neural networks.",
    "Data science involves extracting insights from data using algorithms and statistics."
]

# Text preprocessing
def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

processed_questions = [preprocess(q) for q in questions]

# TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_questions)

# Page title
st.title("AI Chatbot")
st.write("Ask me questions about AI, Machine Learning, or Data Science.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Type your message...")

if user_input:

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Process user input
    processed_input = preprocess(user_input)
    user_vector = vectorizer.transform([processed_input])

    similarity = cosine_similarity(user_vector, X)
    index = np.argmax(similarity)

    bot_response = answers[index]

    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(bot_response)

    # Save bot response
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
