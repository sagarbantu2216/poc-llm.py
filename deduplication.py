from nltk.tokenize import word_tokenize

def dataDeduplication(cleaned_data_to_be_deduplicated):
    print('Data Deduplication Process Started...')
    print('Data Deduplication Process is in progress...')
    
    original_content = []
    duplicate_content = []
    unique_lines = set()
    duplicate_lines = set()
    
    # Split the cleaned data into lines
    lines = cleaned_data_to_be_deduplicated.split('\n')
    
    # Iterate through each line
    for line in lines:
        clean_line = line.strip()  # Remove leading/trailing whitespace and newline characters
        if clean_line not in unique_lines:
            unique_lines.add(clean_line)
            original_content.append(clean_line)
        else:
            duplicate_lines.add(clean_line)
            duplicate_content.append(clean_line)
    
    print('Data Deduplication Process Completed...')
    
    # Calculate word counts
    def calculate_word_count(text):
        tokens = word_tokenize(text)
        return len(tokens)

    # Calculate word count of original content
    original_content_text = '\n'.join(original_content)
    original_word_count = calculate_word_count(original_content_text)
    print(f"Word count of original content: {original_word_count}")
    
    # Calculate word count of duplicate content
    duplicate_content_text = '\n'.join(duplicate_content)
    duplicate_word_count = calculate_word_count(duplicate_content_text)
    print(f"Word count of duplicate content: {duplicate_word_count}")

    
    return original_content_text, duplicate_content_text, original_word_count, duplicate_word_count
