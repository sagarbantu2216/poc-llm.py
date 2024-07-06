#Importing necessary libraries

from langchain.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
# from langchain_community.vectorstores import FAISS

# Function to create a vector store from text chunks
def get_vectorstore(text_chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    # vectorstore = Chroma.from_documents(documents=text_chunks, embedding=embeddings)
    vectorstore = Chroma.from_documents(documents=text_chunks, embedding=embeddings, persist_directory="db2")
    db = vectorstore
    # print(db.get().keys())
    # print("1",len(db.get()["ids"]))
    # print("2",db.get()['embeddings'])
    # print("3",db.get()['metadatas'])
    # print("4",db.get()['documents'])
    # print("5",db.get()['uris'])
    # print("6",db.get()['data'])
    # print("7",db.get()['included'])
    # Print the list of source files
    # for x in range(len(db.get()["metadatas"])):
    #     # print(db.get()["metadatas"][x])
    #     doc = db.get()["metadatas"][x]
    #     source = doc["source"]
        
    #     print(source)
    return vectorstore


# # Function to create a vector store from text chunks
# def get_vectorstore(text_chunks):
#     embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
#     # vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

#     return vectorstore