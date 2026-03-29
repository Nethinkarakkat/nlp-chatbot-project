import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load semantic model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Knowledge base
knowledge_base = {
    "what is artificial intelligence": "Artificial Intelligence enables machines to simulate human intelligence.",
    "applications of artificial intelligence": "AI is used in healthcare, finance, robotics, recommendation systems, and self-driving cars.",
    "advantages of artificial intelligence": "AI improves efficiency, automates repetitive tasks, and helps analyze large datasets.",
    "disadvantages of artificial intelligence": "AI systems can be expensive, require large datasets, and may introduce bias.",
    
    "what is machine learning": "Machine learning is a subset of AI that allows systems to learn from data.",
    "types of machine learning": "The main types are supervised learning, unsupervised learning, and reinforcement learning.",
    
    "what is deep learning": "Deep learning is a machine learning technique based on neural networks with multiple layers.",
    
    "what is nlp": "Natural Language Processing enables computers to understand human language.",
    "applications of nlp": "NLP is used in chatbots, translation systems, sentiment analysis, and voice assistants.",
    
    "what is data science": "Data science focuses on extracting insights from data using statistics and machine learning.",
    
    "who created python": "Python was created by Guido van Rossum in 1991.",
    
    "what is a chatbot": "A chatbot is a program that simulates conversation with users using natural language."
}

questions = list(knowledge_base.keys())
answers = list(knowledge_base.values())

# Convert questions to embeddings
question_embeddings = model.encode(questions)

# Streamlit UI
st.title("AI Chatbot using NLP")
st.write("Ask me questions about AI, Machine Learning, NLP, or Data Science.")

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

# User input
user_input = st.chat_input("Type your message")

if user_input:

    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    text = user_input.lower()

    # Greeting detection
    if text in ["hi", "hello", "hey"]:
        response = "Hello! 😊 Ask me anything about AI, Machine Learning, NLP, or Data Science."

    elif "thank" in text:
        response = "You're welcome! Feel free to ask more questions."

    elif "bye" in text:
        response = "Goodbye! Have a great day."

    else:
        # Semantic similarity
        user_embedding = model.encode([user_input])
        similarity = cosine_similarity(user_embedding, question_embeddings)

        best_match = np.argmax(similarity)

        if similarity[0][best_match] < 0.4:
            response = "I'm not sure about that. Try asking about AI, Machine Learning, NLP, or Data Science."
        else:
            response = answers[best_match]

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
