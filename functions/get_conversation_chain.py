#Importing necessary libraries
from langchain.memory import ConversationBufferMemory # remove and use inmemory if possible
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the ChatGroq language model
llm = ChatGroq(groq_api_key=groq_api_key, model='Llama3-70b-8192', temperature=0.1, max_tokens=2000)


# Function to create a conversation chain
def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key='chat_history',output_key='answer', return_messages=True)
    retriever=vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 2})
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        # retriever=vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 2}),
        retriever = retriever,
        return_source_documents=True,
        memory=memory
        
    )
    return conversation_chain, retriever

# #Function for retrieving the information from conversation chain

# def get_conversation_chain(vectorstore):
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#     conversation_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )

#     return conversation_chain