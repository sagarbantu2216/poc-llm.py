#Importing necessary libraries

from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings

#Importing functions written in another files
from functions.extract_text_from_documents import get_documents_from_files
from functions.get_text_chunks import get_text_chunks
from functions.get_vectorstore import get_vectorstore
from functions.get_conversation_chain import get_conversation_chain
from deduplication import dataDeduplication 
from functions.chrono_res_process_hyperlinks import extract_sections
from functions.chrono_res_process_hyperlinks import convert_to_hyperlinks
import json

# Suppress specific warning
warnings.filterwarnings("ignore", message="`resume_download` is deprecated and will be removed in version 1.0.0.")

app = Flask(__name__)
CORS(app)

#Initializing the global variables
conversation_chain = None
retriever = None

# app route for uploading files
@app.route('/upload', methods=['POST'])
def upload_files():
    uploaded_files = request.files.getlist('files')
    if not uploaded_files:
        return jsonify({'error': 'No files provided'}), 400
    
    global raw_documents, vectorstore, conversation_chain, retriever
    
    raw_documents = get_documents_from_files(uploaded_files)
    text_chunks = get_text_chunks(raw_documents)
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain, retriever = get_conversation_chain(vectorstore)
    
    
    return jsonify({'message': 'Files processed successfully'}), 200



#app route for asking question
@app.route('/ask', methods=['POST'])
def ask_question():
    user_question = request.json.get('query')
    print("-------------")
    print(user_question)

    global conversation_chain
    if conversation_chain is None:
        return jsonify({'error': 'Conversation chain not initialized'}), 500

    response = conversation_chain({'question': user_question})
    # print(response)
    # context = vectorstore.similarity_search(user_question)
    # print(context)
    # print(context[0].page_content)
    # docs = context[1].metadata['source']
    # print(docs)
    
    return jsonify({'result': response['answer']}), 200

#app route for deduplication functionality
@app.route('/deduplicate', methods=['POST'])
def deduplicate_data():
    global raw_documents
    if not raw_documents:
        return jsonify({'error': 'No text data to deduplicate'}), 400

    raw_text = "".join([doc.page_content for doc in raw_documents])
    deduplicated_text, duplicate_text, original_word_count, duplicate_word_count = dataDeduplication(raw_text)
    return jsonify({
        'deduplicated_text': deduplicated_text,
        'duplicate_text': duplicate_text,
        'original_word_count': original_word_count,
        'duplicate_word_count': duplicate_word_count
    }), 200
    
#app route for chronological order functionality
@app.route('/chronological_order', methods=['POST'])
def arrange_chronologically():
    section_names = [
        "Problem List",
        "Medical History",
        "Medications",
        "Social History",
        "Surgical History",
        "Family History",
        "Vital Signs",
        "Lab Results",
        "Imaging Results",
        "Pathology Reports"
    ]

    res_ans_list = [] # list to store the response['answers'] from the model
    source_documents_list = []  # List to store filtered source document paths
    for section_name in section_names:
        query = f"""Can you organize the entries within the EHR data for {section_name} chronologically 
        to facilitate easy access and understanding,The data should be organized into the following sections
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
        with the most recent item listed first
        Please provide at least 10 entries per section and also for each entry please show the date as well?"""

        global conversation_chain
        if conversation_chain is None:
            return jsonify({'error': 'Conversation chain not initialized'}), 500
        
        def filter_metadata(source):
            if source.endswith('.pdf') or source.endswith('.txt'):
                return source
            else:
                return None
            
        response = conversation_chain({'question': query})
        res_ans_list.append(response['answer'])
        # print(response)
        # print(type(response))
        # print(response['source_documents'])
        md_source = response['source_documents'][1].metadata['source']
        print(md_source)
        filtered_source = filter_metadata(md_source)
        source_documents_list.append(filtered_source)

        
        print(source_documents_list)
    
        print("**********************")
        # last_metadata = response["source_documents"]["metadata"]
        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++") 
        # print(json.dumps(last_metadata, indent=4))
        # print(response.metadata['source'])
        
        # print(retriever.get_relevant_documents(query))
        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # Assuming 'vectorstore.similarity_search' returns the context for each query
        # context = vectorstore.similarity_search(query)
        # print("**********************")
        # print(context)
        # print("**********************")
        # for i in response:
        #     print(i.metadata.get('source'))
        
        # pageContent = context[0].page_content  # Assuming you process this data accordingly
        # results[section_name] = pageContent
        # results1[section_name] = response['answer']
        # results[section_name] = pageContent
        # print("********************")
        # print(pageContent, "\n\n\n")
        # print("********************")
    # print('results..................',results)
    return jsonify({'answer':res_ans_list, 'source documents': source_documents_list}),200


#app route for chronological order functionality
# @app.route('/chronological_order', methods=['POST'])
# def arrange_chronologically():
#     query = """Can you organize the entries within the EHR data chronologically to facilitate easy access and understanding? 
#             The data should be organized into the following sections, with the most recent item listed first in each section:
#             Problem List
#             Medical History
#             Medications
#             Social History
#             Surgical History
#             Family History
#             Vital Signs
#             Lab Results
#             Imaging Results
#             Pathology Reports
#             Please provide at least 10 entries per section?"""
    
#     global conversation_chain
#     if conversation_chain is None:
#         return jsonify({'error': 'Conversation chain not initialized'}), 500

#     response = conversation_chain({'query': query})
#     print(response['result'])
#     context = vectorstore.similarity_search(query)
#     print(context)
#     pageContent = context[0].page_content
#     print("********************")
#     print(pageContent)

#     # def extract_sections(response):
#     #     lines = response['result'].split("\n")
#     #     sections = {}
#     #     current_section = None
        
#     #     for line in lines:
#     #         if line.startswith("**") and line.endswith("**"):
#     #             current_section = line.strip("**").strip()
#     #             sections[current_section] = []
#     #         elif current_section and line.strip():
#     #             sections[current_section].append(line.strip())
        
#     #     return sections

#     # sections = extract_sections(response)

#     # def convert_to_hyperlinks(entries):
#     #     hyperlink_template = '<a href="#" onclick="return false;">{}</a>'
#     #     hyperlinks = [hyperlink_template.format(entry) for entry in entries]
#     #     return hyperlinks

#     # html_sections = {}
#     # for section, entries in sections.items():
#     #     hyperlinks = convert_to_hyperlinks(entries)
#     #     hyperlinks_html = "<br>".join(hyperlinks)
#     #     html_sections[section] = hyperlinks_html

#     # html_template = "<html><body>"
#     # for section, content in html_sections.items():
#     #     html_template += f"<h3>{section}</h3><div>{content}</div><br>"
#     # html_template += "</body></html>"

#     # return jsonify({'html_template': html_template}), 200
#     return jsonify(response['result']), 200

@app.route('/summarize', methods=['POST'])
def summarize_data():
    query = """Can you provide concise summaries of consult notes to give healthcare providers key insights at a glance? 
            Please arrange the consult notes in chronological order, with the most recent consult note listed first?"""
    
    global conversation_chain
    if conversation_chain is None:
        return jsonify({'error': 'Conversation chain not initialized'}), 500

    response = conversation_chain({'question': query})
    print(response)
    # context = vectorstore.similarity_search(query)
    # print(context)
    # print(context[0].page_content)
    # docs = context[1].metadata['source']
    # print(docs)
    return jsonify({'result':response['answer']}), 200
    # return jsonify(response), 200

if __name__ == '__main__':
 app.run(host='0.0.0.0', port=5002, debug=True)