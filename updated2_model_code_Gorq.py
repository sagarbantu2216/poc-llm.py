import streamlit as st

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from htmlTemplates import css, bot_template, user_template
import chardet
# from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
import warnings
import os

# Suppress specific warning
warnings.filterwarnings("ignore", message="`resume_download` is deprecated and will be removed in version 1.0.0.")

# Set the page configuration
st.set_page_config(page_title="Chat with PDFs and Text Documents", page_icon=":books:")

from dotenv import load_dotenv
load_dotenv()
## Load the Groq and Google API key from the .env file
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the Ollama-llama3 language model
# llm = ChatGroq(groq_api_key =groq_api_key, model='Gemma-7b-it')
llm = ChatGroq(groq_api_key =groq_api_key, model='Llama3-70b-8192')
# llm = ChatGroq(groq_api_key =groq_api_key, model='Mixtral-8x7b-32768')


# Function to extract text from PDFs or text documents
def get_text_from_documents(docs):
    text = ""
    for doc in docs:
        if doc.name.endswith('.pdf'):
            pdf_reader = PdfReader(doc)
            for page in pdf_reader.pages:
                text += page.extract_text()
        else:
            # Detect encoding
            raw_data = doc.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            text += raw_data.decode(encoding)
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create a conversation chain
def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to handle user input
def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process the documents first.")
        return

    response = st.session_state.conversation({'query': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            

# Main function for Streamlit app
def main():
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Chat with PDFs and Text Documents :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
    
    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader("Upload PDFs or Text Documents", type=["pdf", "txt"], accept_multiple_files=True)
        if docs:
            raw_text = get_text_from_documents(docs)
            # st.write(raw_text)
            with st.spinner("Processing"):
                st.write("Text extraction successful!")
                
                text_chunks = get_text_chunks(raw_text)
                st.write("Text chunks created!")
                
                vectorstore = get_vectorstore(text_chunks)
                st.write("VectorStore DB Ready!Go Ahead and ask questions")
                
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == "__main__":
    main()
