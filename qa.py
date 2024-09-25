# create an api from 'makersuite.google.com' and store it in any python file.
# pip install google.generativeai
# pip install streamlit
import google.generativeai as genai
import streamlit as st
import os
from google_api_secret import api_key

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')

chat=model.start_chat(history=[])

def get_gemini_response(question):
    response=chat.send_message(question,stream=True)
    return response

# initializing streamlit
st.set_page_config(page_title="Gemini LLM Application")
st.header("Q&A Chatbot")

# initializing session state for chat history if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state["chat_history"]=[]

input=st.text_input("input:",key="input")
submit=st.button("Submit The Question")

if submit and input:
    response=get_gemini_response(input)
    # add user question and answer to session chat history
    st.session_state["chat_history"].append(("you",input))
    st.subheader("Answer")
    for chunk in response:
        st.write(chunk.text)
        st.session_state["chat_history"].append(("Bot",chunk.text))
st.subheader("See the Chat History..")  

for role,text in st.session_state["chat_history"]:
    st.write(f"{role}:{text}")
# to run the above code in terminal: go to file then choose new then choose terminal and then type streamlit run qa.py and press enter