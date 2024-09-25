import google.generativeai as genai
import streamlit as st
import os
from google_api_secret import api_key

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')

chat=model.start_chat()

def get_gemini_response(question):
    response=chat.send_message(question)
    return response

# initilizing steamlit
st.set_page_config(page_title="Gemini LLM Application2")
st.header("Chat now..")

input=st.text_input("input:",key="input")
submit=st.button("Submit The Question")

if submit and input:
    response=get_gemini_response(input)
    st.subheader("Answer")
    for chunk in response:
        st.write(chunk.text)