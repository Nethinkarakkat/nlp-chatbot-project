import streamlit as st
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
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
    "Machine Learning is a subset of AI that allows systems to learn from data.",
    "Natural Language Processing allows computers to understand human language.",
    "A chatbot is a program that simulates conversation with users.",
    "Python was created by Guido van Rossum.",
    "Deep learning is a type of machine learning based on neural networks.",
    "Data science involves extracting insights from data using algorithms and statistics."
]

# Preprocess text
def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

processed_questions = [preprocess(q) for q in questions]

# TF-IDF model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_questions)

# Streamlit UI
st.title("AI Chatbot")

st.write("You can ask me questions about AI, Machine Learning, NLP, and Data Science.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! 👋 I'm your AI chatbot. How can I help you today?"
    })

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Type your message...")

if user_input:

    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    user_lower = user_input.lower()

    # Greeting detection
    if user_lower in ["hi", "hello", "hey"]:
        response = "Hello! 😊 What would you like to know about AI or Machine Learning?"

    elif "thanks" in user_lower or "thank you" in user_lower:
        response = "You're welcome! Let me know if you have more questions."

    else:
        # NLP similarity matching
        processed_input = preprocess(user_input)
        user_vector = vectorizer.transform([processed_input])
        similarity = cosine_similarity(user_vector, X)

        index = np.argmax(similarity)

        if similarity[0][index] < 0.25:
            response = "I'm not sure about that. Try asking about AI, ML, NLP, or Data Science."
        else:
            response = answers[index]

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
