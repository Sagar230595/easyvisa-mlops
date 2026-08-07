import json
import re
import os
from fpdf import FPDF

# Paths
transcript_path = r"C:\Users\spadh\.gemini\antigravity\brain\40aaa5dd-688c-4ec1-89c0-664406badc49\.system_generated\logs\transcript_full.jsonl"
pdf_path = "Conversation_History.pdf"

class PDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 12)
        self.cell(0, 10, "EasyVisa MLOps Conversation History", border=False, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

    def chapter_title(self, role):
        self.set_font("helvetica", "B", 12)
        if role == "User":
            self.set_text_color(0, 102, 204) # Blue
        else:
            self.set_text_color(0, 153, 76) # Green
        self.cell(0, 10, f"{role}:", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0) # Black

    def chapter_body(self, text):
        self.set_font("helvetica", "", 10)
        # Handle unicode by encoding to ascii and ignoring errors to prevent FPDF crash
        safe_text = text.encode('ascii', 'ignore').decode('ascii')
        self.multi_cell(0, 5, safe_text)
        self.ln(5)

def clean_user_input(text):
    # Extract only the content within <USER_REQUEST> tags if they exist
    match = re.search(r'<USER_REQUEST>(.*?)</USER_REQUEST>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def main():
    pdf = PDF()
    pdf.add_page()
    
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                
                # We only want user inputs and model text responses
                step_type = data.get("type")
                source = data.get("source")
                content = data.get("content", "")
                
                if not content:
                    continue
                    
                if step_type == "USER_INPUT":
                    cleaned = clean_user_input(content)
                    if cleaned and not cleaned.startswith("{{ CHECKPOINT"):
                        pdf.chapter_title("User")
                        pdf.chapter_body(cleaned)
                        
                elif step_type == "PLANNER_RESPONSE" and source == "MODEL":
                    # For model, we just print the text content, ignore tool calls
                    if content and not content.startswith("CRITICAL INSTRUCTION"):
                        pdf.chapter_title("Assistant")
                        pdf.chapter_body(content.strip())
                        
        pdf.output(pdf_path)
        print("PDF generated successfully.")
    except Exception as e:
        print(f"Error generating PDF: {e}")

if __name__ == "__main__":
    main()
