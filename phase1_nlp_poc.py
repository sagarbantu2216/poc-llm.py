from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
from functions.extract_docuements_from_files import get_documents_from_files
from functions.get_text_chunks import get_text_chunks
from functions.get_vectorstore import get_vectorstore
from functions.get_conversation_chain import get_conversation_chain
from deduplication import dataDeduplication 
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import XMLOutputParser
from sections import create_section_dict
from sentenceoffset import process_document
import os
import json
# Suppress specific warning
warnings.filterwarnings("ignore", message="`resume_download` is deprecated and will be removed in version 1.0.0.")
app = Flask(__name__)
CORS(app)

conversation_chain = None #Initializing the global variables
retriever = None
vectorstore = None
raw_documents = []
text_chunks = None
response = ''
md_source = ''
QA_source_documents = []

def clear_vectorstore(): # Function to clear the vector store
    global vectorstore
    vectorstore = None
    print("Vector store cleared.")
    
def clear_conversation_chain():  # Function to clear conversation memory
    global conversation_chain, retriever
    conversation_chain = None
    retriever = None
    print("Conversation chain and retriever cleared.")
    
# app route for uploading files
@app.route('/upload', methods=['POST'])
def upload_files():
    global raw_documents, text_chunks, vectorstore, conversation_chain, new_doc_ids
    raw_documents = []
    text_chunks = None
    clear_vectorstore()  # Clear the existing vector store
    clear_conversation_chain()  # Clear the conversation chain
    conversation_chain = None
    new_doc_ids = []  # Clear the new_doc_ids list
    uploaded_files = request.files.getlist('files')
    if not uploaded_files:
        return jsonify({'error': 'No files provided'}), 400
    raw_documents = get_documents_from_files(uploaded_files)
    text_chunks = get_text_chunks(raw_documents)
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain = get_conversation_chain(vectorstore)
    new_doc_ids = [doc.metadata['doc_id'] for doc in raw_documents]
    return jsonify({'message': 'Files processed successfully'}), 200

uploadFolder = r'C:\Edvenswa\RAG POC - June\RAG + LLM + GORQ\execution_clientPOC\copied_files' # Define the upload folder

#app route for retrieving the information from uploaded files 
@app.route('/get-file-content', methods=['POST'])
def get_file_content():
    print(f"Current Uploaded folder: {uploadFolder}")
    filename = request.args.get('filename')
    if not filename:
        print("Error: No filename provided in the request.") 
        return jsonify({'error': 'No filename provided'}), 400
    file_path = os.path.join(uploadFolder, filename)    
    if os.path.exists(file_path):
        print(f"File found. Preparing to read: {file_path}")  # Debug print before reading the file
        with open(file_path, 'r') as f:
            raw_text = f.read()
        print("File read successfully. Sending content back.")  # Confirm successful file read
        return jsonify({'text': raw_text}), 200
    else:
        print(f"File not found: {file_path}")  # Additional debug print
        return jsonify({'error': 'File not found'}), 404

#app route for asking question
@app.route('/ask', methods=['POST'])
def ask_question():
    user_question = request.json.get('query')
    print(user_question)
    global conversation_chain, new_doc_ids
    if conversation_chain is None:
        return jsonify({'error': 'Conversation chain not initialized'}), 500
    response = conversation_chain({'question': user_question})
    filtered_docs = [doc for doc in response['source_documents'] if doc.metadata['doc_id'] in new_doc_ids]
    if filtered_docs:
        md_source = filtered_docs[0].metadata['source']
        source_filename = os.path.basename(md_source)
    else:
        source_filename = "No document matches the criteria"
    return jsonify({'response': response['answer'] ,'source_document': source_filename}), 200

#app route for asking question
@app.route('/nlp', methods=['POST'])
def nlp_process():
    user_question = request.json.get('query')
    print(user_question)
    file_path = r"C:\Users\RajasimhaG\Downloads\NLP_Attributes(sections).csv"
    sections_json = create_section_dict(file_path) # Create a JSON of section names and their corresponding section ids
    sections_dict = json.loads(sections_json)   # Load the JSON string to a dictionary
    # global raw_documents
    # section_name = ["Medications"]
    # sentenceoffset_info = process_document(raw_documents[0].page_content, section_name)
    # print("process documents logic calling passing raw_documents and section name")
    # print(json.dumps(sentenceoffset_info, indent=4))
    global conversation_chain, new_doc_ids
    if conversation_chain is None:
        return jsonify({'error': 'Conversation chain not initialized'}), 500
    # Create a mapping of section names to OIDs
    section_name_to_oid = {section['section_name']: section['oid'] for section in sections_dict}
    prompt = """You are a medical assistant AI with access to a patient's medical records. 
                Use the patient's medical records to provide a detailed and accurate response. 
                Format your response according to the following JSON schema:
                {
                    "header": "string",
                    "originalText": "string",
                    "age": "numeric value, 'unknown' if null",
                    "dob": "MM/DD/YYYY, 'unknown' if null",
                    "gender": "male or 'female', 'unknown' if null",
                    "race": "American Indian or Alaska Native, Asian, Black or African American, Native Hawaiian or Other Pacific Islander, White, Other, 'unknown' if null",
                    "ethnicity": "Hispanic or Latino, Not Hispanic or Latino, 'unknown' if null",
                    "smokingStatus": "SMOKER, CURRENT_SMOKER, PAST_SMOKER, NON_SMOKER, UNKNOWN, FORMER_SMOKER, 'unknown' if null",
                    "result": [
                        {
                            "name": "DiseaseDisorderMention, LabMention, MedicationMention, ProcedureMention, SignSymptomMention, SectionHeader, gender, AnatomicalSiteMention, EntityMention, MedicalDeviceMention, BacteriumMention, GeneMention",
                            "sectionOid": "see sections sheet, 'SIMPLE_SEGMENT' if null",
                            "sectionName": "see sections sheet, omitted if 'SIMPLE_SEGMENT'",
                            "sectionOffset": "character offset for the entire section",
                            "sentence": "character offset for the sentence",
                            "extendedSentence": "character offset for the extended sentence",
                            "text": "annotated text with character offsets",
                            "attributes": {
                                "derivedGeneric": "1 - derived generic, 0 - not derived generic",
                                "polarity": "positive, negated, default positive",
                                "relTime": "current status, history status, family history status, probably status, default current status",
                                "date": "MM-DD-YYYY, omitted if null",
                                "status": "stable, unstable, controlled, not controlled, deteriorating, getting worse, improving, resolved, resolving, unresolved, uncontrolled, worsening, well-controlled, unchanged, chronic, diminished, new diagnosis, omitted if null, expandable list",
                                "medDosage": "MedicationMention attribute",
                                "medForm": "MedicationMention attribute",
                                "medFrequencyNumber": "MedicationMention attribute",
                                "medFrequencyUnit": "MedicationMention attribute",
                                "medRoute": "MedicationMention attribute",
                                "medStrengthNum": "MedicationMention attribute",
                                "medStrengthUnit": "MedicationMention attribute",
                                "labUnit": "LabMention attribute",
                                "labValue": "LabMention attribute",
                                "umlsConcept": [
                                    {
                                        "codingScheme": "UMLS Vocabulary associated with UMLS Atom",
                                        "cui": "UMLS CUI appropriate for annotation under 'text'",
                                        "tui": "UMLS TUI",
                                        "code": "Code associated with UMLS Atom",
                                        "preferredText": "Preferred text description of UMLS Atom"
                                    }
                                ]
                            },
                            "sectionOffset": "character offset for the entire section",
                            "sentence": "character offset for the sentence",
                            "extendedSentence": "character offset for the extended sentence"
                        }
                    ]
                }
                For the section Problem List, Medications from the patient document, please extract relevant information 
                from the patient's records and provide detailed information. Include the onset, duration, 
                diagnosis date, status, associated symptoms, dosage, frequency, route of administration, 
                duration of use, and any relevant details. 
                Additionally, provide the sectionOffset, sentence, and extendedSentence for each result. 
                The sectionOffset should indicate the character offset for the entire section. 
                The sentence should indicate the character offset for the specific sentence. 
                The extendedSentence should indicate the character offset for the extended sentence.
                can you generate all the coding schemas mentioned below for each information extracted from the patient's records.
                Refer to this URL for UMLS concept codes: https://uts-ws.nlm.nih.gov. 
                Use ICD10CM, SNOMED, RXNORM as coding schemas for diseases and medications. 
                Ensure the response is accurate and structured according to the provided schema.
                """
    response = conversation_chain.invoke({'question': prompt})
    filtered_docs = [doc for doc in response['source_documents'] if doc.metadata['doc_id'] in new_doc_ids]
    if filtered_docs:
        md_source = filtered_docs[0].metadata['source']
        source_filename = os.path.basename(md_source)
    else:
        source_filename = "No document matches the criteria"
    parser = JsonOutputParser()
    json_response = parser.parse(response['answer'])
    print(json_response)
    for result in json_response['result']:     # Map the section names to their OIDs in the response
        section_name = result['sectionName']
        if section_name in section_name_to_oid:
            result['sectionOid'] = section_name_to_oid[section_name]
    # for sentence_info in sentenceoffset_info:     # Find the matching text in sentenceoffset_info and append the offsets
    #     if result['text'] in sentence_info['text']:
    #         result['sectionOffset'] = sentence_info['sectionOffset']
    #         result['sentence'] = sentence_info['sentence']
    #         result['extendedSentence'] = sentence_info['extendedSentence']
    #         break
    return jsonify({'response': json_response, 'source document': source_filename},), 200

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
                    "Problem List","Medical History","Medications","Social History","Surgical History",
                    "Family History","Vital Signs","Lab Results","Imaging Results","Pathology Reports"
                    ]
    res_ans_dict = {} # list to store the response['answers'] from the model
    global new_doc_ids  # Reference the global new_doc_ids variable
    for section_name in section_names:
        query = f"""Given a collection of electronic health records (EHR) stored as documents, I am working on 
                    organizing the entries within the EHR data for the section "{section_name}" chronologically,
                    to facilitate easy access and understanding. The goal is to list the most recent entries first.
                    For this section, please provide a structured summary that includes at least 10 entries, 
                    if available. Each entry must be accompanied by a date to indicate when it was recorded or 
                    updated. This organization is crucial for creating a clear, chronological narrative of the patient's 
                    health history based on the available data.Additionally, it is important to identify and include 
                    the source document's name (either a PDF or a text file) from which each entry is derived. 
                    This will aid in tracing the information back to its original context if needed.Could you 
                    assist by processing the uploaded documents, extracting the relevant information for 
                    "{section_name}", and organizing it as requested? Please ensure to maintain accuracy and 
                    clarity in the chronological arrangement and presentation of the data."""
        global conversation_chain
        if conversation_chain is None:
            return jsonify({'error': 'Conversation chain not initialized'}), 500
        response = conversation_chain({'question': query})  
        filtered_response = {'answer': '', 'source_documents': []}  
        for doc in response['source_documents']:
            if doc.metadata['doc_id'] in new_doc_ids:
                filtered_response['source_documents'].append(doc)
                if not filtered_response['answer']:
                    filtered_response['answer'] = response['answer']

        if filtered_response['source_documents']:
            md_source = filtered_response['source_documents'][0].metadata['source']
            src_doc_filename = os.path.basename(md_source)
            res_ans_dict[section_name] = {
                'answer': filtered_response['answer'],
                'document_id': filtered_response['source_documents'][0].metadata['doc_id'],
                'source_document': src_doc_filename
            }    
    return jsonify({'result': res_ans_dict}), 200

#app route for summarization functionality
@app.route('/summarize', methods=['POST'])
def summarize_data():
    query = """I have a collection of consult notes from various documents. I need you to generate concise 
               summaries for each consult note. These summaries should provide healthcare providers with key 
               insights quickly. Please ensure to:
                - Highlight crucial information such as diagnoses, treatment plans, 
                  and significant patient details.
                - Arrange the summaries in chronological order, with the most recent consult note at the top.
                The goal is to capture the essence of each consult, offering a clear view of the 
                patient's current status or medical opinions. Can you process the documents to extract and 
                summarize the consult notes as described, ensuring the summaries are informative and presented 
                with the latest information first? If you need more details to proceed, please let me know."""
    global conversation_chain, new_doc_ids
    if conversation_chain is None:
        return jsonify({'error': 'Conversation chain not initialized'}), 500
    response = conversation_chain({'question': query})
    filtered_docs = [doc for doc in response['source_documents'] if doc.metadata['doc_id'] in new_doc_ids]
    if filtered_docs:
        md_source = filtered_docs[0].metadata['source']
        summary_source_filename = os.path.basename(md_source)
    else:
        summary_source_filename = "No document matches the criteria"
    return jsonify({'result':response['answer'], 'source_document':summary_source_filename}), 200

if __name__ == '__main__':
    app.run(debug=False, port= 8000)
