import os 
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
import os
from openai import OpenAI

# st.sidebar.image('file.png')

client = OpenAI(api_key = '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')

def run_llm(prompt) : 

    chat_completion = client.chat.completions.create(
        messages = [
            {
                'role' : 'user' , 
                'content' : prompt
            }
        ] , model = 'gpt-4o')

    return chat_completion.choices[0].message.content

def check_prompt(prompt) : 

    try : 
        prompt.replace('' , '')
        return True 
    except : return False

os.environ['PINECONE_API_KEY'] = '5546964f-7996-445c-a4c4-44df700cd7d7'

embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')

vc = PineconeVectorStore(index_name = 'test' , embedding = embeddings)

def run_rag(query , vc) : 

    similar_docs = vc.similarity_search(query , k = 5)
    context = '\n'.join([doc.page_content for doc in similar_docs])

    prompt = open('prompt.txt').read().format(context , query)

    response = run_llm(prompt)

    return response

def check_mesaage() : 
    '''
    Function to check the messages
    '''

    if 'messages' not in st.session_state : st.session_state.messages = []

check_mesaage()

for message in st.session_state.messages : 

    with st.chat_message(message['role']) : st.markdown(message['content'])

query = st.chat_input('Ask me anything')

if check_prompt(query) :

    with st.chat_message('user'): st.markdown(query)

    st.session_state.messages.append({
        'role' : 'user' , 
        'content' : query
    })

    if query != None or query != '' : 

        response = run_rag(query , vc)

        with st.chat_message('assistant') : st.markdown(response)

        st.session_state.messages.append({
            'role' : 'assistant' , 
            'content' : response
        })
