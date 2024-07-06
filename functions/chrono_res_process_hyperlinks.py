#Function for extracting sections from the chronological response
def extract_sections(response):
        lines = response['result'].split("\n")
        sections = {}
        current_section = None
        
        for line in lines:
            if line.startswith("**") and line.endswith("**"):
                current_section = line.strip("**").strip()
                sections[current_section] = []
            elif current_section and line.strip():
                sections[current_section].append(line.strip())
        
        return sections
    
#Function for converting each section entries to hyperlinks
def convert_to_hyperlinks(entries):
        hyperlink_template = '<a href="#" onclick="return false;">{}</a>'
        hyperlinks = [hyperlink_template.format(entry) for entry in entries]
        return hyperlinks