import os
import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2
import requests
from bs4 import BeautifulSoup
import textwrap
import json
import logging

# --- Logging Setup ---
logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# --- Gemini API Setup ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# --- Utility Functions ---
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        if os.path.getsize(pdf_path) > 2 * 1024 * 1024:  # 2MB limit
            return "Error: File size exceeds 2MB."
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        logging.info("PDF extraction successful")
    except FileNotFoundError:
        logging.error("PDF file not found")
        return "Error: PDF file not found."
    except Exception as e:
        logging.error(f"Error while processing PDF: {e}")
        return f"Error while processing PDF: {e}"
    return text

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        text_parts = soup.find_all('p')
        logging.info("URL extraction successful")
        return '\n'.join([p.get_text() for p in text_parts])
    except Exception as e:
        logging.error(f"Error while fetching URL: {e}")
        return f"Error while fetching URL: {e}"

def summarize_with_gemini(text, length="short", format="paragraph"):
    try:
        chat = model.start_chat()
        if format == "bullets":
            prompt = (
                f"Summarize the following text in a {length} style. "
                f"Use bullet points – one idea per line:\n\n{text}"
            )
        else:
            prompt = f"Summarize the following text in a {length} style as a paragraph:\n\n{text}"
        response = chat.send_message(prompt)
        logging.info("Summary generated")
        return response.text
    except Exception as e:
        logging.error(f"Error while summarizing with Gemini: {e}")
        return f"Error while summarizing with Gemini: {e}"

def format_paragraph(text, width=100):
    return "\n".join(textwrap.wrap(text, width=width))

# --- New Functions per Mission ---
def extract_entities_with_gemini(text):
    try:
        chat = model.start_chat()
        prompt = (
            "Extract the following entities from the text below:\n\n"
            "- Names of people\n- Dates\n- Locations\n- Organizations\n\n"
            "Return the result as a JSON object with the following keys:\n"
            "{ 'names': [], 'dates': [], 'locations': [], 'organizations': [] }\n\n"
            f"Text:\n{text}"
        )
        response = chat.send_message(prompt)
        logging.info("Entities extracted")
        return response.text
    except Exception as e:
        logging.error(f"Error while extracting entities: {e}")
        return f"Error while extracting entities: {e}"

def extract_tables_with_gemini(text):
    try:
        chat = model.start_chat()
        prompt = (
            "If the following text contains any tables or lists, extract them and represent them in JSON format.\n\n"
            f"Text:\n{text}"
        )
        response = chat.send_message(prompt)
        logging.info("Tables extracted")
        return response.text
    except Exception as e:
        logging.error(f"Error while extracting tables/lists: {e}")
        return f"Error while extracting tables/lists: {e}"

def answer_questions_with_gemini(text, questions):
    try:
        chat = model.start_chat()
        prompt = (
            "Based only on the following document content, answer these questions. "
            "If the answer cannot be found, say 'Information not available in the document.' "
            "For each answer, provide a confidence score (High/Medium/Low).\n\n"
            f"Document:\n{text}\n\nQuestions:\n{questions}"
        )
        response = chat.send_message(prompt)
        logging.info("Questions answered")
        return response.text
    except Exception as e:
        logging.error(f"Error while answering questions: {e}")
        return f"Error while answering questions: {e}"

def interactive_questioning(text):
    try:
        chat = model.start_chat()
        print("\nInteractive mode: Ask follow-up questions about the document. Type 'exit' to stop.")
        while True:
            user_question = input("Your question: ")
            if user_question.lower() == 'exit':
                break
            prompt = (
                "Based only on the following document content, answer the user's question. "
                "If the answer cannot be found, say 'Information not available in the document.' "
                "Also provide a confidence score (High/Medium/Low).\n\n"
                f"Document:\n{text}\n\nQuestion:\n{user_question}"
            )
            response = chat.send_message(prompt)
            print("\nAnswer:")
            print(response.text)
    except Exception as e:
        logging.error(f"Error in interactive questioning: {e}")
        print(f"Error: {e}")

def compare_documents(text1, text2):
    try:
        chat = model.start_chat()
        prompt = (
            "Compare the following two documents. Provide similarities, differences, and a comparison summary.\n\n"
            f"Document 1:\n{text1}\n\nDocument 2:\n{text2}"
        )
        response = chat.send_message(prompt)
        logging.info("Documents compared")
        return response.text
    except Exception as e:
        logging.error(f"Error while comparing documents: {e}")
        return f"Error while comparing documents: {e}"

def transform_style_with_gemini(text, style, audience):
    try:
        chat = model.start_chat()
        prompt = (
            f"Rewrite the following text in a {style} style, targeting {audience}:\n\n{text}"
        )
        response = chat.send_message(prompt)
        logging.info("Style transformed")
        return response.text
    except Exception as e:
        logging.error(f"Error while transforming style: {e}")
        return f"Error while transforming style: {e}"

def expand_content_with_gemini(text, section):
    try:
        chat = model.start_chat()
        prompt = (
            f"Expand the following section of the document:\n\n{section}\n\n"
            "Provide more detail and explanation."
        )
        response = chat.send_message(prompt)
        logging.info("Content expanded")
        return response.text
    except Exception as e:
        logging.error(f"Error while expanding content: {e}")
        return f"Error while expanding content: {e}"

def translate_content_with_gemini(text, target_language):
    try:
        chat = model.start_chat()
        prompt = (
            f"Translate the following text to {target_language}. "
            "In addition to translation, adapt the text to fit the cultural context of the target language audience. "
            "Highlight any localization changes or adaptations made (idioms, phrases, examples) by marking them clearly.\n\n"
            f"Text:\n{text}"
        )
        response = chat.send_message(prompt)
        logging.info("Content translated")
        return response.text
    except Exception as e:
        logging.error(f"Error while translating content: {e}")
        return f"Error while translating content: {e}"

def recommend_content_with_gemini(text, user_profile):
    try:
        chat = model.start_chat()
        prompt = (
            f"User profile: {user_profile}\n\n"
            f"Document content:\n{text}\n\n"
            "Based on the user's profile and the document's content, provide a personalized content recommendation. "
            "Explain why this document is or is not a good match for the user."
        )
        response = chat.send_message(prompt)
        logging.info("Personalized recommendation generated")
        return response.text
    except Exception as e:
        logging.error(f"Error while generating recommendation: {e}")
        return f"Error while generating recommendation: {e}"

def generate_code_documentation(code_snippet):
    try:
        chat = model.start_chat()
        prompt = (
            "Analyze the following code and generate clear documentation. "
            "Explain the functionality, list the key functions/classes/variables, and provide examples of usage.\n\n"
            f"Code:\n{code_snippet}"
        )
        response = chat.send_message(prompt)
        logging.info("Code documentation generated")
        return response.text
    except Exception as e:
        logging.error(f"Error while generating code documentation: {e}")
        return f"Error while generating code documentation: {e}"

def generate_quiz_with_gemini(text):
    try:
        chat = model.start_chat()
        prompt = (
            "Based on the following document, create a short quiz (3-5 questions) with multiple choice answers.\n\n"
            f"Document:\n{text}"
        )
        response = chat.send_message(prompt)
        logging.info("Quiz generated")
        return response.text
    except Exception as e:
        logging.error(f"Error while generating quiz: {e}")
        return f"Error while generating quiz: {e}"

# --- Main Program ---
def main():
    source_type = input("Enter 'file' to upload a document or 'url' to use a web address: ").lower()
    text = ""

    if source_type == 'file':
        file_path = input("Enter the path to the file (PDF or TXT): ").strip().strip('"')
        if file_path.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif file_path.lower().endswith('.txt'):
            try:
                if os.path.getsize(file_path) > 2 * 1024 * 1024:
                    text = "Error: File size exceeds 2MB."
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    logging.info("TXT file read")
            except Exception as e:
                logging.error(f"Error while reading TXT file: {e}")
                text = f"Error while reading TXT file: {e}"
        else:
            text = "Error: Unsupported file format."

    elif source_type == 'url':
        url = input("Enter the web address: ").strip().strip('"')
        text = extract_text_from_url(url)
    else:
        print("Invalid input.")
        return

    if text and not text.startswith("Error"):
        print("\nExtracted text:")
        lines = text.splitlines()
        if len(lines) > 10:
            for line in lines[:10]:
                print(line)
            print("...")
        else:
            print(text)

        while True:
            print("\nWhat would you like to do?")
            print("1 - Summarize text")
            print("2 - Extract entities")
            print("3 - Extract tables/lists")
            print("4 - Answer questions")
            print("5 - Interactive questioning")
            print("6 - Compare with another document")
            print("7 - Transform style")
            print("8 - Expand content")
            print("9 - Translate + localize content")
            print("10 - Personalized content recommendation")
            print("11 - Generate code documentation")
            print("12 - Generate quiz")
            print("13 - Exit")

            choice = input("Enter your choice (1-13): ")

            if choice == '1':
                summary_length = input("Enter summary length (short / medium / detailed): ").lower()
                summary_format = input("Enter summary format (paragraph / bullets): ").lower()
                summary = summarize_with_gemini(text, length=summary_length, format=summary_format)
                print("\nSummary:")
                print(format_paragraph(summary))

            elif choice == '2':
                entities = extract_entities_with_gemini(text)
                print("\nExtracted Entities:")
                print(entities)

            elif choice == '3':
                tables = extract_tables_with_gemini(text)
                print("\nExtracted Tables/Lists:")
                print(tables)

            elif choice == '4':
                questions = input("Enter your questions separated by semicolons (;): ")
                answers = answer_questions_with_gemini(text, questions)
                print("\nAnswers:")
                print(answers)

            elif choice == '5':
                interactive_questioning(text)

            elif choice == '6':
                print("\n--- Upload second document for comparison ---")
                source2 = input("Enter 'file' or 'url' for second document: ").lower()
                text2 = ""
                if source2 == 'file':
                    file_path2 = input("Enter the path to the file (PDF or TXT): ").strip().strip('"')
                    if file_path2.lower().endswith('.pdf'):
                        text2 = extract_text_from_pdf(file_path2)
                    elif file_path2.lower().endswith('.txt'):
                        with open(file_path2, 'r', encoding='utf-8') as f:
                            text2 = f.read()
                    else:
                        print("Error: Unsupported file format.")
                        continue
                elif source2 == 'url':
                    url2 = input("Enter the web address: ").strip().strip('"')
                    text2 = extract_text_from_url(url2)
                else:
                    print("Invalid input.")
                    continue

                comparison = compare_documents(text, text2)
                print("\nComparison Result:")
                print(comparison)

            elif choice == '7':
                style = input("Enter target style (formal, informal, poetic, journalistic): ")
                audience = input("Enter target audience (children, students, professionals, etc.): ")
                transformed = transform_style_with_gemini(text, style, audience)
                print("\nTransformed Content:")
                print(transformed)

            elif choice == '8':
                section = input("Enter section text to expand: ")
                expanded = expand_content_with_gemini(text, section)
                print("\nExpanded Content:")
                print(expanded)

            elif choice == '9':
                target_language = input("Enter target language (e.g., French, Arabic, Hebrew): ")
                translated = translate_content_with_gemini(text, target_language)
                print("\nTranslated Content:")
                print(translated)

            elif choice == '10':
                user_profile = input("Enter user profile (interests, preferences, background): ")
                recommendation = recommend_content_with_gemini(text, user_profile)
                print("\nRecommendation:")
                print(recommendation)

            elif choice == '11':
                code_snippet = input("Enter your code snippet: ")
                documentation = generate_code_documentation(code_snippet)
                print("\nGenerated Documentation:")
                print(documentation)

            elif choice == '12':
                quiz = generate_quiz_with_gemini(text)
                print("\nGenerated Quiz:")
                print(quiz)

            elif choice == '13':
                print("Exiting the program.")
                break

            else:
                print("Invalid choice.")
    else:
        print(text)

# Entry point
if __name__ == "__main__":
    main()
