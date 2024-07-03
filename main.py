import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory # remove and 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from docx import Document
import xml.etree.ElementTree as ET
from htmlTemplates import css, bot_template, user_template
import os
import warnings
import tempfile
import chardet
from deduplication import dataDeduplication  # Import your deduplication function

# Suppress specific warning
warnings.filterwarnings("ignore", message="`resume_download` is deprecated and will be removed in version 1.0.0.")

# Set the page configuration
st.set_page_config(page_title="Chat with Documents", page_icon=":books:")

from dotenv import load_dotenv
load_dotenv()
## Load the Groq and Google API key from the .env file
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the ChatGroq language model
llm = ChatGroq(groq_api_key=groq_api_key, model='Llama3-70b-8192', temperature=0.1, max_tokens= 2000) # add temperature into this

# Function to extract text from PDFs or text documents using LangChain document loaders
def get_text_from_documents(docs):
    text = ""
    
    for doc in docs:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(doc.read())
            tmp_file.flush()
            tmp_file.seek(0)
            
            if doc.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_file.name)
                documents = loader.load()
                for document in documents:
                    text += document.page_content
            elif doc.name.endswith('.txt'):
                with open(tmp_file.name, 'rb') as f:
                    raw_data = f.read()
                encoding = chardet.detect(raw_data)['encoding']
                loader = TextLoader(tmp_file.name, encoding=encoding)
                documents = loader.load()
                for document in documents:
                    text += document.page_content
            elif doc.name.endswith('.xml'):
                tree = ET.parse(tmp_file.name)
                root = tree.getroot()
                for elem in root.iter():
                    if elem.text:
                        text += elem.text + "\n"
            elif doc.name.endswith('.docx'):
                doc = Document(tmp_file.name)
                for para in doc.paragraphs:
                    text += para.text + "\n"
            else:
                continue
    
    return text

# Function to split text into chunks 1000 by 100
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
    st.session_state.chat_history = [{"role": "user", "content": user_question},
                                     {"role": "assistant", "content": response['result']}]
    st.session_state.user_input_response = response['result']
    
    # st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}", response['result']), unsafe_allow_html=True)
    

# Function to handle deduplication
def handle_deduplication():
    if "raw_text" not in st.session_state:
        st.warning("Please upload and process the documents first.")
        return
    if st.session_state.conversation is None:
        st.warning("Please upload and process the documents first.")
        return

    deduplicated_text, duplicate_text, original_word_count, duplicate_word_count = dataDeduplication(st.session_state.raw_text)
    st.session_state.deduplicated_text = deduplicated_text
    st.session_state.duplicate_text = duplicate_text
    st.session_state.original_word_count = original_word_count
    st.session_state.duplicate_word_count = duplicate_word_count

# Function to handle deduplication
def handle_datadeduplication(query):
    if st.session_state.conversation is None:
        st.warning("Please upload and process the documents first.")
        return

    # st.session_state.chat_history.append({"role": "user", "content": query})
    response = st.session_state.conversation({'query': query})
    # Clear chat history and keep only the current query and response
    st.session_state.chat_history = [{"role": "user", "content": query},
                                     {"role": "assistant", "content": response['result']}]
    
    # st.session_state.chat_history.append({"role": "assistant", "content": response['result']})
    st.session_state.deduplication_response = response['result']  # Save the response for download

# Function to handle Chronological Order Arrangement
def handle_chronological_order(query):
    if st.session_state.conversation is None:
        st.warning("Please upload and process the documents first.")
        return

    # st.session_state.chat_history.append({"role": "user", "content": query})
    response = st.session_state.conversation({'query': query})
    # Clear chat history and keep only the current query and response
    st.session_state.chat_history = [{"role": "user", "content": query},
                                     {"role": "assistant", "content": response['result']}]
    # st.session_state.chat_history.append({"role": "assistant", "content": response['result']})
    st.session_state.chronological_response = response['result']  # Save the response for download

# Function to handle summarization
def handle_summarization(query):
    if st.session_state.conversation is None:
        st.warning("Please upload and process the documents first.")
        return

    # st.session_state.chat_history.append({"role": "user", "content": query})
    response = st.session_state.conversation({'query': query})
    # Clear chat history and keep only the current query and response
    st.session_state.chat_history = [{"role": "user", "content": query},
                                     {"role": "assistant", "content": response['result']}]
    # st.session_state.chat_history.append({"role": "assistant", "content": response['result']})
    st.session_state.summarization_response = response['result']  # Save the response for download

# # Function to display chat history on the main UI
def display_chat_history():
    for i, message in enumerate(st.session_state.chat_history):
        if message['role'] == "assistant":  # Display only assistant messages
            st.write(bot_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
            if i == len(st.session_state.chat_history) - 1:
                # Add a download button for the latest bot message
                st.download_button(
                    label="Download Response",
                    data=message['content'],
                    file_name=f"response_{i//2 + 1}.txt",
                    mime="text/plain",
                key=f"download_button_{i}"
                )

# Main function for Streamlit app
def main():
    st.write(css, unsafe_allow_html=True)
    # st.image("edvenswalogo.jfif", width=100)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.header("Chat with Documents :books:")
    
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
    
    with st.sidebar:
        st.subheader("Upload your documents here")
        docs = st.file_uploader("Upload PDFs, Text, XML, or DOCX Documents", type=["pdf", "txt", "xml", "docx"], accept_multiple_files=True)
        
        # Buttons for additional functionalities
        if st.button("Deduplication"):
            query = "Can you remove duplicate entries from the EHR data to ensure that each entry is unique?"
            handle_datadeduplication(query)
            handle_deduplication()
            if "deduplicated_text" in st.session_state:
                st.download_button(
                    label="Download Deduplicated Text",
                    data=st.session_state.deduplicated_text,
                    file_name="deduplicated_text.txt",
                    mime="text/plain",
                    key="download_deduplication"
                )
                
        if st.button("Chronological Order"):
            query = """Can you organize the entries within the EHR data chronologically to facilitate easy access and understanding? 
            The data should be organized into the following sections, with the most recent item listed first in each section:
            Problem List
            Medical History
            Medications
            Social History
            Surgical History
            Family History
            Vital Signs
            Lab Results
            Imaging Results
            Pathology Reports
            Please provide at least 10 entries per section. Along with this can you provide me the list of uploaded sources file or documents from where LLM model is retrieving the responses? """
            handle_chronological_order(query)
            if "chronological_response" in st.session_state:
                st.download_button(
                    label="Download Chronological Order Response",
                    data=st.session_state.chronological_response,
                    file_name="chronological_order_response.txt",
                    mime="text/plain",
                    key="download_chronological"
                )
        if st.button("Summarization"):
            query = """Can you provide concise summaries of consult notes to give healthcare providers key insights at a glance? 
            Please arrange the consult notes in chronological order, with the most recent consult note listed first?"""
            handle_summarization(query)
            if "summarization_response" in st.session_state:
                st.download_button(
                    label="Download Summarization Response",
                    data=st.session_state.summarization_response,
                    file_name="summarization_response.txt",
                    mime="text/plain",
                    key="download_summarization"
                )
        
    if docs:
        # Check if the text extraction has already been done
        if 'raw_text' not in st.session_state:
            raw_text = get_text_from_documents(docs)
            st.session_state.raw_text = raw_text

            with st.spinner("Processing"):
                with st.sidebar:
                    st.write("Text extraction successfully completed and Merged into a single text!")
                
                text_chunks = get_text_chunks(raw_text)
                st.session_state.text_chunks = text_chunks
                with st.sidebar:
                    st.write("Text chunks created!")
                
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.vectorstore = vectorstore
                with st.sidebar:
                    st.write("VectorStore DB Ready! Go ahead and ask questions.")
                
                st.session_state.conversation = get_conversation_chain(vectorstore)
                
                #Display message on the main UI after processing
                st.success("**Documents processed successfully! Go ahead and ask questions**")
        else:
            with st.sidebar:
                st.write("Documents already processed. Go ahead and ask questions.")
    
    # Display chat history on the main UI
    display_chat_history()

if __name__ == "__main__":
    main()
