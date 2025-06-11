import os
from google.ai.generativelanguage import GenerativeServiceClient
from dotenv import load_dotenv
import PyPDF2
import requests
from bs4 import BeautifulSoup
import streamlit as st

# --- Gemini API Setup ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# --- Utility Functions ---
def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        if uploaded_file.size > 2 * 1024 * 1024:
            return "Error: File size exceeds 2MB."
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    except Exception as e:
        return f"Error while processing PDF: {e}"
    return text

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        text_parts = soup.find_all('p')
        return '\n'.join([p.get_text() for p in text_parts])
    except Exception as e:
        return f"Error while fetching URL: {e}"

def gemini_chat(prompt):
    chat = model.start_chat()
    response = chat.send_message(prompt)
    return response.text

# --- Streamlit App ---
st.set_page_config(page_title="Document AI App", layout="wide")
st.title("📄 Document AI Processor with Gemini")

# Document upload or URL input
source = st.radio("Choose document source:", ["Upload File", "Enter URL"])

text = ""

if source == "Upload File":
    uploaded_file = st.file_uploader("Upload PDF or TXT file", type=["pdf", "txt"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".pdf"):
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8")
        else:
            st.error("Unsupported file format.")

elif source == "Enter URL":
    url = st.text_input("Enter the URL:")
    if url:
        text = extract_text_from_url(url)

# Display extracted text
if text and not text.startswith("Error"):
    with st.expander("📄 Show Extracted Text", expanded=False):
        st.write(text[:2000] + ("..." if len(text) > 2000 else ""))

    # Select Action
    st.header("🛠️ Choose an Action")
    action = st.selectbox("Select an action:", [
        "Summarize Text",
        "Extract Entities",
        "Extract Tables/Lists",
        "Answer Questions",
        "Compare with Another Document",
        "Transform Style",
        "Expand Content",
        "Translate + Localize Content",
        "Personalized Content Recommendation",
        "Generate Code Documentation",
        "Generate Quiz"
    ])

    if action == "Summarize Text":
        length = st.selectbox("Summary Length:", ["short", "medium", "detailed"])
        format_style = st.selectbox("Format:", ["paragraph", "bullets"])
        if st.button("Generate Summary"):
            prompt = (
                f"Summarize the following text in a {length} style. "
                f"{'Use bullet points.' if format_style == 'bullets' else 'As a paragraph.'}\n\n{text}"
            )
            st.write(gemini_chat(prompt))

    elif action == "Extract Entities":
        if st.button("Extract Entities"):
            prompt = (
                "Extract the following entities from the text below:\n\n"
                "- Names of people\n- Dates\n- Locations\n- Organizations\n\n"
                "Return the result as a JSON object with the following keys:\n"
                "{ 'names': [], 'dates': [], 'locations': [], 'organizations': [] }\n\n"
                f"Text:\n{text}"
            )
            st.write(gemini_chat(prompt))

    elif action == "Extract Tables/Lists":
        if st.button("Extract Tables/Lists"):
            prompt = (
                "If the following text contains any tables or lists, extract them and represent them in JSON format.\n\n"
                f"Text:\n{text}"
            )
            st.write(gemini_chat(prompt))

    elif action == "Answer Questions":
        questions = st.text_area("Enter your questions (separated by semicolons ';'):")
        if st.button("Answer Questions"):
            prompt = (
                "Based only on the following document content, answer these questions. "
                "If the answer cannot be found, say 'Information not available in the document.' "
                "For each answer, provide a confidence score (High/Medium/Low).\n\n"
                f"Document:\n{text}\n\nQuestions:\n{questions}"
            )
            st.write(gemini_chat(prompt))

    elif action == "Compare with Another Document":
        uploaded_file2 = st.file_uploader("Upload second PDF or TXT file", type=["pdf", "txt"], key="second_doc")
        text2 = ""
        if uploaded_file2:
            if uploaded_file2.name.endswith(".pdf"):
                text2 = extract_text_from_pdf(uploaded_file2)
            elif uploaded_file2.name.endswith(".txt"):
                text2 = uploaded_file2.read().decode("utf-8")
            if st.button("Compare Documents"):
                prompt = (
                    "Compare the following two documents. Provide similarities, differences, and a comparison summary.\n\n"
                    f"Document 1:\n{text}\n\nDocument 2:\n{text2}"
                )
                st.write(gemini_chat(prompt))

    elif action == "Transform Style":
        style = st.text_input("Target style (formal, informal, poetic, journalistic):")
        audience = st.text_input("Target audience (children, students, professionals, etc.):")
        if st.button("Transform Style"):
            prompt = (
                f"Rewrite the following text in a {style} style, targeting {audience}:\n\n{text}"
            )
            st.write(gemini_chat(prompt))

    elif action == "Expand Content":
        section = st.text_area("Enter section text to expand:")
        if st.button("Expand Content"):
            prompt = (
                f"Expand the following section of the document:\n\n{section}\n\n"
                "Provide more detail and explanation."
            )
            st.write(gemini_chat(prompt))

    elif action == "Translate + Localize Content":
        target_language = st.text_input("Target language (e.g., French, Arabic, Hebrew):")
        if st.button("Translate + Localize"):
            prompt = (
                f"Translate the following text to {target_language}. "
                "In addition to translation, adapt the text to fit the cultural context of the target language audience. "
                "Highlight any localization changes or adaptations made (idioms, phrases, examples) by marking them clearly.\n\n"
                f"Text:\n{text}"
            )
            st.write(gemini_chat(prompt))

    elif action == "Personalized Content Recommendation":
        user_profile = st.text_area("Enter user profile (interests, preferences, background):")
        if st.button("Generate Recommendation"):
            prompt = (
                f"User profile: {user_profile}\n\n"
                f"Document content:\n{text}\n\n"
                "Based on the user's profile and the document's content, provide a personalized content recommendation. "
                "Explain why this document is or is not a good match for the user."
            )
            st.write(gemini_chat(prompt))

    elif action == "Generate Code Documentation":
        code_snippet = st.text_area("Enter your code snippet:")
        if st.button("Generate Code Documentation"):
            prompt = (
                "Analyze the following code and generate clear documentation. "
                "Explain the functionality, list the key functions/classes/variables, and provide examples of usage.\n\n"
                f"Code:\n{code_snippet}"
            )
            st.write(gemini_chat(prompt))

    elif action == "Generate Quiz":
        if st.button("Generate Quiz"):
            prompt = (
                "Based on the following document, create a short quiz (3-5 questions) with multiple choice answers.\n\n"
                f"Document:\n{text}"
            )
            st.write(gemini_chat(prompt))
else:
    if source == "Upload File" and uploaded_file is None:
        st.info("Please upload a file to proceed.")
    elif source == "Enter URL" and not url:
        st.info("Please enter a URL to proceed.")
