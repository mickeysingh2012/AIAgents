import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import os
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Load AI model from Ollama
llm = OllamaLLM(model="mistral")

# Initialize memory in Streamlit session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

# Prompt template
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Previous conversation: {chat_history}\nUser: {question}\nAI:"
)

# Function to run AI response with memory
def run_chain(question):
    chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in st.session_state.chat_history.messages])
    response = llm.invoke(prompt.format(chat_history=chat_history_text, question=question))
    st.session_state.chat_history.add_user_message(question)
    st.session_state.chat_history.add_ai_message(response)
    return response

# Speech recognition
recognizer = sr.Recognizer()

# Function to speak using gTTS
def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    #adjust speed
    os.system("ffmpeg -y -i response.mp3 -filter:a 'atempo=1.2' fast_response.mp3")
    os.system("mpg123 fast_response.mp3")
# Function to listen using mic
def listen():
    with sr.Microphone() as source:
        print("\nListening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        print(f"You said: {query}")
        return query.lower()
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand.")
        return ""
    except sr.RequestError:
        print("Speech Recognition is unavailable.")
        return ""

# --- Streamlit UI ---
st.title("AI Voice Assistant (Web UI)")
st.write("Click the button below to speak to your AI assistant!")

if st.button("Start Listening"):
    user_query = listen()
    if user_query:
        ai_response = run_chain(user_query)
        st.write(f"**You**: {user_query}")
        st.write(f"**AI**: {ai_response}")
        speak(ai_response)

# Show chat history
st.subheader("Chat History")
for msg in st.session_state.chat_history.messages:
    st.write(f"**{msg.type.capitalize()}**: {msg.content}")
