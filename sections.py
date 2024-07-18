import pandas as pd
import json

def create_section_dict(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, header=None)
    
    # Initialize an empty list
    sections_list = []
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Split the row into parts
        parts = row[0].split(',')
        if len(parts) >= 3:
            section_info = {
                "section_name": parts[2].strip(),
                "loinc_code": parts[1].strip(),
                "oid": parts[0].strip()
            }
            sections_list.append(section_info)
    
    # Get the length of the sections list
    num_sections = len(sections_list)
    # print(f"Number of sections in the list: {num_sections}")
    # Convert the list to JSON
    sections_json = json.dumps(sections_list, indent=4)
    
    return sections_json

# # Optionally, write the JSON to a file
# with open("sections.json", "w") as json_file:
#     json_file.write(sections_json)
