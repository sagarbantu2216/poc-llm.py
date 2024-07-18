from langchain.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.vectorstores import Chroma

#Function to create a vector store from text chunks
def get_vectorstore(text_chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    vectorstore = Chroma.from_documents(documents=text_chunks, embedding=embeddings)
    return vectorstore