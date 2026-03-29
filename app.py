import streamlit as st
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    "Natural Language Processing enables computers to understand human language.",
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

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_questions)

# UI
st.title("AI Chatbot")
st.write("You can ask me questions about AI, Machine Learning, NLP, and Data Science.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! 👋 I'm your AI chatbot. How can I help you today?"
    })

# Show conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type your message...")

if user_input:

    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    text = user_input.lower()

    # Greeting
    if text in ["hi", "hello", "hey"]:
        response = "Hello! 😊 Ask me anything about AI, Machine Learning, or NLP."

    # AI questions
    elif "ai" in text or "artificial intelligence" in text:
        response = "Artificial Intelligence enables machines to simulate human intelligence."

    # Machine learning
    elif "machine learning" in text or "ml" in text:
        response = "Machine Learning is a subset of AI that allows systems to learn from data."

    # NLP
    elif "nlp" in text or "natural language processing" in text:
        response = "Natural Language Processing enables computers to understand human language."

    # Chatbot
    elif "chatbot" in text:
        response = "A chatbot is a program that simulates conversation with users."

    # Python
    elif "python" in text:
        response = "Python was created by Guido van Rossum."

    # Thanks
    elif "thanks" in text or "thank you" in text:
        response = "You're welcome! 😊"

    else:
        # fallback to similarity
        processed_input = preprocess(user_input)
        user_vector = vectorizer.transform([processed_input])

        similarity = cosine_similarity(user_vector, X)
        index = np.argmax(similarity)

        if similarity[0][index] < 0.25:
            response = "I'm not sure about that. Try asking about AI, Machine Learning, NLP, or Data Science."
        else:
            response = answers[index]

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
