import streamlit as st
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLP resources
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Knowledge base questions
questions = [
    "what is artificial intelligence",
    "what is machine learning",
    "what is natural language processing",
    "what is chatbot",
    "who created python",
    "what is deep learning",
    "what is data science"
]

# Corresponding responses
answers = [
    "Artificial Intelligence enables machines to simulate human intelligence.",
    "Machine Learning is a subset of AI that allows systems to learn from data.",
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

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_questions)

# Streamlit UI
st.title("NLP Chatbot")
st.write("Ask questions about AI, ML, or Data Science.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
prompt = st.chat_input("Ask your question")

if prompt:

    st.chat_message("user").write(prompt)

    # Preprocess input
    user_processed = preprocess(prompt)

    # Convert to vector
    user_vector = vectorizer.transform([user_processed])

    # Compute similarity
    similarity = cosine_similarity(user_vector, X)

    index = np.argmax(similarity)

    response = answers[index]

    st.chat_message("assistant").write(response)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})