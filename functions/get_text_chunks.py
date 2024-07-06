#Importing Necessary Libraries

from langchain_text_splitters import RecursiveCharacterTextSplitter


# Function to split documents into text chunks
def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# # Function to split text into chunks
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=100,
#         length_function=len,
#         is_separator_regex=False
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks