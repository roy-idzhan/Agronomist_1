import os 
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
import os
from openai import OpenAI

# Initialize OpenAI client with API key from Streamlit secrets
client = OpenAI(api_key=st.secrets['OPENAI'])

# URL of your logo image
logo_url = "https://imgur.com/a/ZpyZ58y"  # Replace with your actual image URL

# Display the logo at the top of the chatbot interface
st.image(logo_url, width=200)  # You can adjust the width as needed

# Add a welcome note
if 'welcome_shown' not in st.session_state:
    st.markdown("### Welcome to the FarmByte Ai Wizard!")
    st.markdown("I'm here to assist you with all your agronomy-related queries. Feel free to ask me anything.")
    st.session_state.welcome_shown = True  # Set flag to avoid showing the welcome message again


# Function to run the LLM with a given prompt
def run_llm(prompt): 
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'user', 
                'content': prompt
            }
        ], 
        model='gpt-4o-mini'
    )
    return chat_completion.choices[0].message.content

# Function to check if the prompt is valid
def check_prompt(prompt): 
    try: 
        prompt.replace('', '')
        return True 
    except: 
        return False

# Set the Pinecone API key as an environment variable
os.environ['PINECONE_API_KEY'] = '5546964f-7996-445c-a4c4-44df700cd7d7'

# Initialize the HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# Create a Pinecone vector store instance
vc = PineconeVectorStore(index_name='test', embedding=embeddings)

# Function to run the RAG (Retrieval-Augmented Generation) model
def run_rag(query, vc): 
    similar_docs = vc.similarity_search(query, k=5)
    context = '\n'.join([doc.page_content for doc in similar_docs])

    # Include memory of past interactions
    memory = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])

    # Combine memory, context, and query into the prompt
    prompt = open('prompt.txt').read().format(memory + "\n\n" + context, query)

    # Get the LLM response
    response = run_llm(prompt)

    return response

# Function to check if the 'messages' key exists in session state
def check_mesaage(): 
    '''
    Function to check the messages
    '''
    if 'messages' not in st.session_state: 
        st.session_state.messages = []

check_mesaage()

# Display previous chat messages
for message in st.session_state.messages: 
    with st.chat_message(message['role']): 
        st.markdown(message['content'])

# Get new user input
query = st.chat_input('Ask me anything')

# If the query is valid, process it
if check_prompt(query):
    with st.chat_message('user'): 
        st.markdown(query)

    st.session_state.messages.append({
        'role': 'user', 
        'content': query
    })

    if query is not None and query != '': 
        response = run_rag(query, vc)

        with st.chat_message('assistant'): 
            st.markdown(response)

        st.session_state.messages.append({
            'role': 'assistant', 
            'content': response
        })

# st.session_state.messages.append({
#     'role': 'assistant', 
#     'content': response
# })  # This line is commented out as per your request
