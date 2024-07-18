import json

def extract_sections_with_offsets(document_text, section_keywords):
    """Extract sections from document text based on section keywords and track offsets."""
    sections = {}
    current_section = None
    current_section_offset = None
    position = 0
    for line in document_text.split('\n'):
        # Check if the line is a section title
        if any(keyword in line for keyword in section_keywords):
            current_section = line
            current_section_offset = position
            sections[current_section] = {'offset': current_section_offset, 'content': []}
        elif current_section:
            sections[current_section]['content'].append((position, line))
        position += len(line) + 1  # +1 for the newline character
    return sections

def extract_sentences_with_offsets(document_text):
    """Extract sentences from document text and track offsets."""
    sentences = []
    position = 0
    start_position = 0
    for i, char in enumerate(document_text):
        if char == '.':
            sentence = document_text[start_position:i+1].strip()
            if sentence:
                sentences.append({'offset': (start_position, i+1), 'sentence': sentence})
            start_position = i + 1
    return sentences

def process_document(document_text, section_keywords):
    """Process document to extract sections, sentences, and their offsets."""
    sections = extract_sections_with_offsets(document_text, section_keywords)
    sentences = extract_sentences_with_offsets(document_text)
    result = []
    for section_title, section_data in sections.items():
        section_offset = section_data['offset']
        section_content = section_data['content']
        section_text = "\n".join(line for _, line in section_content)
        section_end_offset = section_offset + len(section_text)
        for sentence_data in sentences:
            sentence_offset = sentence_data['offset']
            sentence_text = sentence_data['sentence']
            
            # Check if the sentence is within the section
            if section_offset <= sentence_offset[0] < section_end_offset:
                extended_sentence_offset = (sentence_offset[0], sentence_offset[1])
                
                result.append({
                    "section": section_title,
                    "sectionOffset": [section_offset, section_end_offset],
                    "sentence": sentence_offset,
                    "extendedSentence": extended_sentence_offset,
                    "text": sentence_text
                })

    return result