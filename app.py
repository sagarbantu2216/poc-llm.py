from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from docx import Document
import xml.etree.ElementTree as ET
import os
import tempfile
import chardet
from deduplication import dataDeduplication
from dotenv import load_dotenv
import warnings

# Suppress specific warning
warnings.filterwarnings("ignore", message="`resume_download` is deprecated and will be removed in version 1.0.0.")

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load the Groq API key from the .env file
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Initialize the ChatGroq language model
llm = ChatGroq(groq_api_key=groq_api_key, model='Llama3-70b-8192', temperature=0.1, max_tokens=2000)

# Global variable to store conversation chain
conversation_chain = None

# Function to extract text from PDFs or text documents using LangChain document loaders
def get_text_from_documents(docs):
    text = ""
    
    for doc in docs:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(doc.read())
            tmp_file.flush()
            tmp_file.seek(0)
            
            if doc.filename.endswith('.pdf'):
                loader = PyPDFLoader(tmp_file.name)
                documents = loader.load()
                for document in documents:
                    text += document.page_content
            elif doc.filename.endswith('.txt'):
                with open(tmp_file.name, 'rb') as f:
                    raw_data = f.read()
                encoding = chardet.detect(raw_data)['encoding']
                loader = TextLoader(tmp_file.name, encoding=encoding)
                documents = loader.load()
                for document in documents:
                    text += document.page_content
            elif doc.filename.endswith('.xml'):
                tree = ET.parse(tmp_file.name)
                root = tree.getroot()
                for elem in root.iter():
                    if elem.text:
                        text += elem.text + "\n"
            elif doc.filename.endswith('.docx'):
                doc = Document(tmp_file.name)
                for para in doc.paragraphs:
                    text += para.text + "\n"
            else:
                continue
        os.remove(tmp_file.name)
    
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

# API endpoint to handle file uploads and process documents
@app.route('/upload', methods=['POST'])
def upload_documents():
    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "No files provided"}), 400
    
    raw_text = get_text_from_documents(files)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    global conversation_chain
    conversation_chain = get_conversation_chain(vectorstore)
    return jsonify({"message": "Documents processed successfully", "raw_text": raw_text})

# API endpoint to handle user questions
@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400
    response = conversation_chain({'query': question})
    return jsonify({"response": response['result']})

# API endpoint for deduplication
@app.route('/deduplicate', methods=['POST'])
def deduplicate_text():
    data = request.get_json()
    raw_text = data.get('raw_text')
    if not raw_text:
        return jsonify({"error": "No text provided"}), 400

    deduplicated_text, duplicate_text, original_word_count, duplicate_word_count = dataDeduplication(raw_text)
    return jsonify({
        "deduplicated_text": deduplicated_text,
        "duplicate_text": duplicate_text,
        "original_word_count": original_word_count,
        "duplicate_word_count": duplicate_word_count
    })

if __name__ == "__main__":
    app.run(debug=True)