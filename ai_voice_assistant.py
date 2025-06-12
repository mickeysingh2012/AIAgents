import speech_recognition as sr
from gtts import gTTS
import os
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

#using Google speach for ai voice using gtts & os
#load AI model from Ollama
llm = OllamaLLM(model="mistral")

#Initialize Memory #(LangChain v1.0+)
chat_history = ChatMessageHistory() #store user-AI conversation history

#Speech Recognition
recognizer = sr.Recognizer()

#Function to Speak
def speak(text):
    print(f"AI: {text}")
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    os.system("mpg123 response.mp3")

# Function to Listen
def listen():
    with sr.Microphone() as source:
        print("\nListening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        print(f"You Said: {query}")
        return query.lower()
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand. Try again!")
        return ""
    except sr.RequestError:
        print("Speech Recognition Service Unavailable")
        return ""

# Define AI Chat Prompt
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Previous conversation:\n{chat_history}\nUser: {question}\nAI:"
)

# Function to run AI chat with memory
def run_chain(question):
    chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history.messages])
    response = llm.invoke(prompt.format(chat_history=chat_history_text, question=question))
    chat_history.add_user_message(question)
    chat_history.add_ai_message(response)
    return response

# Main Loop
speak("Hello! I am your AI Assistant. How can I help you today?")
while True:
    query = listen()
    if "exit" in query or "stop" in query:
        speak("Goodbye! Have a great day.")
        break
    if query:
        response = run_chain(query)
        speak(response)
