import os
import chardet
from langchain.document_loaders import PyPDFLoader, TextLoader
from uuid import uuid4

uploadFolder = 'uploaded_files' # Define the upload folder

# Function for removing the directory UPLOAD_FOLDER
def remove_directory(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            os.remove(file_path)
        for name in dirs:
            dir_path = os.path.join(root, name)
            os.rmdir(dir_path)
    os.rmdir(directory)
    
# Function call for printing each document along with the doc_id from the list of documents
def print_documents(documents):
    for document in documents:
        doc_id = document.metadata.get("doc_id", None)
        source = document.metadata.get("source", None)
        # print(f"Document ID: {document.metadata['doc_id']}, Content: {document.page_content}")
        print(f"Document ID: {doc_id}, Source: {source}")
        
# Function for getting the documents from the uploaded files
def get_documents_from_files(docs):
    if os.path.exists(uploadFolder):
        remove_directory(uploadFolder)
    os.makedirs(uploadFolder)
    documents = []
    for doc in docs: # loop through the documents
        doc_id = str(uuid4())
        file_path = os.path.join(uploadFolder, doc.filename)
        with open(file_path, 'wb') as f: # save the document to the upload folder
            f.write(doc.read())
        if doc.filename.endswith('.pdf'):   # check the file type
            loader = PyPDFLoader(file_path)
            loaded_documents = loader.load_and_split()
            for document in loaded_documents:  # add the doc_id to each document's metadata
                document.metadata["doc_id"] = doc_id
            documents.extend(loaded_documents)
        elif doc.filename.endswith('.txt'):   # check if the file is a text document
            with open(file_path, 'rb') as f:  # detect the encoding of the text file
                raw_data = f.read()
            encoding = chardet.detect(raw_data)['encoding']
            loader = TextLoader(file_path, encoding=encoding)
            loaded_documents = loader.load()
            for document in loaded_documents:  # add the doc_id to each document's metadata
                document.metadata["doc_id"] = doc_id
            documents.extend(loaded_documents)
        else:
            print(f"Unsupported file type: {doc.filename}. Skipping...")
    print("Completed processing all documents.")
    print_documents(documents)  # Calling the function to print the documents
    return documents


