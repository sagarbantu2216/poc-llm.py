#Function call for printing each chunk along with the doc_id from the list of chunks
# def printing_chunks(chunks):
#     for chunk in chunks:
#         doc_id = chunk.metadata.get("doc_id", None)
#         chunk_text = chunk.page_content
#         print(f"Chunk ID: {doc_id}, Content: {chunk_text}")

#Function for getting text chunks for the documents to be processed adding metadata doc_id
def get_text_chunks(documents):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False
    )
    for document in documents:
        doc_id = document.metadata.get("doc_id", None)
        split_chunks = text_splitter.split_documents([document])
        for chunk in split_chunks:
            chunk.metadata["doc_id"] = doc_id
            chunks.append(chunk)
    print(len(chunks))
    # printing_chunks(chunks) #calling the function to print the chunks
    return chunks
