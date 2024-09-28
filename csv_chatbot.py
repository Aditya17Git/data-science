from langchain.document_loaders.csv_loader import CSVLoader
loader=CSVLoader(file_path="C:\\Users\\my lapi\\Documents\\bcit_faqs.csv",source_column="prompt")
docs=loader.load()

import google.generativeai as genai
import streamlit as st
import os
from google_api_secret import api_key


genai.configure(api_key=api_key)

# pip install InstructorEmbedding
# pip install torch
# pip install -U sentence-transformers==2.2.2
# pip install faiss-cpu==1.7.4 for vector database
import torch
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS


instructor_embeddings=HuggingFaceInstructEmbeddings()

vectordb=FAISS.from_documents(documents=docs,embedding=instructor_embeddings)
    


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

prompt_template="""
Answer the question from the provided context, make sure to provide all the details, if the answer is not in the provided context the just say,
"I don't know the answer, please ask another question", don't provide the wrong answer.

CONTEXT:{context}
QUESTION:{question}

Answer:
"""
llm=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3,api_key=api_key)
prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
chain=load_qa_chain(llm,chain_type="stuff",prompt=prompt)

def user_input(user_question):
    embeddings=instructor_embeddings
    retriever=vectordb.as_retriever()
    docs=retriever.get_relevant_documents(user_question)
    
    response=chain(
       {"input_documents":docs,"question":user_question}
       ,return_only_outputs=True)
    
    return response

# initializing streamlit

st.title("Chat with BCIT WORLD 🙋‍♂️")

question=st.text_input("question")

if question:
    response=user_input(question)
    st.header("Answer:")
    st.write("Reply:",response["output_text"])
    
    





    
