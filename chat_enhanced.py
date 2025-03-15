import streamlit as st
# Set page config must be the first Streamlit command
st.set_page_config(page_title="Enhanced Gemini Chatbot", page_icon="🤖", layout="wide")

import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure API Key
api_key = os.getenv('GOOGLE_API_KEY')
if api_key is None:
    st.error("GOOGLE_API_KEY not found in .env file")
    st.stop()

genai.configure(api_key=api_key)

# Sidebar configurations
st.sidebar.title("Chatbot Settings")

# Model parameters
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 
                              help="Higher values make output more creative but less focused")
top_p = st.sidebar.slider("Top P", 0.0, 1.0, 0.9,
                         help="Nucleus sampling: diversity of responses")
top_k = st.sidebar.slider("Top K", 1, 40, 20,
                         help="Limits the number of tokens considered for each step of text generation")

# Safety settings
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

# Initialize model with parameters
model = genai.GenerativeModel(
    model_name='gemini-1.5-pro',
    generation_config={
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k
    },
    safety_settings=safety_settings
)

st.title("Enhanced Gemini AI Chatbot")

# Initialize session states
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chat' not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

# Chat history management
def save_chat_history():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(st.session_state.chat_history, f)
    return filename

def clear_chat_history():
    st.session_state.chat_history = []
    st.session_state.chat = model.start_chat(history=[])

# Sidebar controls for chat management
if st.sidebar.button("Clear Chat History"):
    clear_chat_history()

if st.sidebar.button("Save Chat History"):
    filename = save_chat_history()
    st.sidebar.success(f"Chat history saved to {filename}")

# System message configuration
system_prompt = st.sidebar.text_area(
    "System Prompt",
    "You are a helpful and knowledgeable AI assistant.",
    help="Set the AI's behavior and context"
)

# Main chat interface
chat_container = st.container()

with chat_container:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(f"{message['content']}")
            if "timestamp" in message:
                st.caption(f"Sent at {message['timestamp']}")

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    try:
        # Prepare context with system prompt
        context = f"{system_prompt}\n\nUser: {user_input}"
        
        # Get response from Gemini
        response = st.session_state.chat.send_message(context)
        
        # Add assistant message to chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response.text,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Display assistant message
        with st.chat_message("assistant"):
            st.write(response.text)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Display chat statistics
st.sidebar.markdown("---")
st.sidebar.subheader("Chat Statistics")
total_messages = len(st.session_state.chat_history)
user_messages = len([m for m in st.session_state.chat_history if m["role"] == "user"])
assistant_messages = len([m for m in st.session_state.chat_history if m["role"] == "assistant"])

st.sidebar.write(f"Total messages: {total_messages}")
st.sidebar.write(f"User messages: {user_messages}")
st.sidebar.write(f"Assistant messages: {assistant_messages}")