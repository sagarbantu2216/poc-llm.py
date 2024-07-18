#Importing necessary libraries
from langchain.memory import ConversationBufferMemory 
from langchain.chains import ConversationalRetrievalChain
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

#list of groq api keys
groq_api_key_list = ["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",
                     "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",
                     "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",
                     "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",
                     "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",
                     "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",
                     "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",
                     "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",
                     "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",
                     "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",
                     "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",
                     "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",
                     "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",
                     "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",
                     "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",
                     "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"    
                    ]

# groq_api_key = os.getenv("GROQ_API_KEY")
# llm = ChatGroq(groq_api_key=groq_api_key, model='Llama3-70b-8192', temperature=0.1, max_tokens=2000) # groq model

# Function to create a conversation chain
def get_conversation_chain(vectorstore):
    memory = None
    retriever = None
    conversation_chain = None
    use_new_groq_api_key() # Function to use a new GROQ API key
    memory = ConversationBufferMemory(memory_key='chat_history',output_key='answer', return_messages=True)
    retriever=vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 2})
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever = retriever,
        return_source_documents=True,
        memory=memory
    )
    return conversation_chain


def use_new_groq_api_key():
    global groq_api_key
    global llm
    if len(groq_api_key_list) > 0:
        groq_api_key = groq_api_key_list.pop()
        llm = ChatGroq(groq_api_key=groq_api_key, model='Llama3-70b-8192', temperature=0.1, max_tokens=3000) # groq model
        return groq_api_key
    else:
        return None
