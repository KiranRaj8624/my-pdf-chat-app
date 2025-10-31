# streamlit_app.py
"""
Interactive PDF Chatbot (Streamlit app using Gemini API)

This application allows a user to upload a PDF. The app extracts the text,
stores it in the session state, and allows the user to ask questions
based on the PDF's content.
"""

import os
import io
import time
import logging
from typing import List, Tuple

import streamlit as st
from PyPDF2 import PdfReader
import nltk

# --- Optional extras for PDF text extraction (OCR) ---
# These are the same as your Flask app.
# We'll install the dependencies in requirements.txt and packages.txt
try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None

# NLTK
nltk.download('punkt', quiet=True)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger("pdf_chat_app_st")

# Load configuration from environment or Streamlit secrets
def get_gemini_api_key():
    """
    Fetches the Gemini API key securely.
    - First, tries st.secrets (for deployed Streamlit apps).
    - Then, falls back to os.environ (for local development).
    """
    if 'GEMINI_API_KEY' in st.secrets:
        return st.secrets['GEMINI_API_KEY']
    return os.environ.get('GEMINI_API_KEY')

ALLOW_OFFLINE = os.environ.get('ALLOW_OFFLINE', '0') == '1'
MODEL_NAME = os.environ.get('GEMINI_MODEL', 'gemini-1.5-flash')

# Configure genai
GEMINI_API_KEY = get_gemini_api_key()
if genai and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        log.info("Configured google.generativeai")
    except Exception as e:
        log.warning("Failed to configure google.generativeai: %s", e)
        genai = None
else:
    if not GEMINI_API_KEY:
        log.info("GEMINI_API_KEY not set. Set it in Streamlit secrets or environment.")
    if genai is None:
        log.info("google-generativeai not installed or failed to import.")


# ----- PDF Extraction Utilities (Copied from your code) -----
# These functions are unchanged.

def extract_text_from_pdf(file_bytes: bytes) -> Tuple[str, List[bytes]]:
    """Extracts text using PyPDF2 and optionally images for OCR."""
    page_images = []
    text_parts = []
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as e:
        log.warning("PyPDF2 failed to read PDF: %s", e)
        return "", []

    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
            if txt.strip():
                text_parts.append(txt)
        except Exception:
            continue

    if not text_parts and convert_from_bytes is not None:
        log.info("No text extracted by PyPDF2. Attempting OCR...")
        try:
            imgs = convert_from_bytes(file_bytes, dpi=200)
            for im in imgs:
                bio = io.BytesIO()
                im.save(bio, format='PNG')
                page_images.append(bio.getvalue())
        except Exception as e:
            log.warning("pdf2image conversion failed: %s", e)

    return "\n".join(text_parts), page_images

def ocr_pdf_images(page_images: List[bytes]) -> str:
    """Performs OCR on a list of page images."""
    if not page_images or Image is None or pytesseract is None:
        return ''
    log.info(f"Performing OCR on {len(page_images)} images...")
    texts = []
    for i, b in enumerate(page_images):
        try:
            img = Image.open(io.BytesIO(b))
            txt = pytesseract.image_to_string(img)
            if txt:
                texts.append(txt)
        except Exception as e:
            log.warning("OCR failed for page image %d: %s", i, e)
    log.info("OCR complete.")
    return "\n".join(texts)

# ----- Model Generation Utilities (Copied from your code) -----
# This function is unchanged and will work perfectly.

FALLBACK_MODELS = ['gemini-1.5-pro', 'gemini-pro']

def gemini_generate(prompt: str, max_output_tokens: int = 1024) -> str:
    """Resilient wrapper: tries configured model(s); falls back to offline stub if allowed."""
    if genai is None:
        if ALLOW_OFFLINE:
            return f"[OFFLINE-STUB] This is an offline answer based on the document."
        log.error("Generative model not configured and offline mode not enabled.")
        return "Error: The generative model is not configured. (Did you set the API key?)"

    candidates = ([MODEL_NAME] if MODEL_NAME else []) + [m for m in FALLBACK_MODELS if m != MODEL_NAME]
    tried = []
    for m in candidates:
        if not m:
            continue
        tried.append(m)
        try:
            model = genai.GenerativeModel(m)
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_output_tokens,
                temperature=0.1
            )
            resp = model.generate_content(prompt, generation_config=generation_config)
            text = getattr(resp, 'text', None)
            if text:
                return text
            log.warning("Model %s returned empty response or was blocked. Response: %s", m, resp)
            return f"Error: Model {m} returned an empty or blocked response."
        except Exception as e:
            log.warning("Model %s failed: %s", m, e)
            continue

    if ALLOW_OFFLINE:
        return f"[OFFLINE-FALLBACK] All models failed (tried {', '.join(tried)})."
    
    log.error("All model attempts failed (tried: %s)", ", ".join(tried))
    return f"Error: All generative models failed. (Tried: {', '.join(tried)})"


def answer_question_from_context(context: str, question: str) -> str:
    """
    Creates a prompt and calls Gemini to answer a question based on context.
    (Unchanged from your code)
    """
    prompt = (
        "You are a helpful assistant. You will be given a document's text as context. "
        "Your task is to answer the user's question based *only* on the provided context.\n"
        "Do not use any external knowledge. Do not make up information.\n"
        "If the answer is not found in the context, state that clearly "
        "(e.g., 'The provided document does not contain information on this topic.').\n"
        "\n--- DOCUMENT CONTEXT ---\n"
        f"{context}"
        "\n--- END OF CONTEXT ---\n"
        "\nUSER QUESTION:\n"
        f"{question}"
    )
    return gemini_generate(prompt, max_output_tokens=1024)


# ----- Streamlit UI and App Logic -----

st.set_page_config(layout="wide")
st.title("📄 Chat with PDF")

# --- Session State Management ---
# This replaces the Flask session
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = None
    st.session_state.filename = None
    st.session_state.messages = []

# --- Helper function to clear state ---
def clear_session_state():
    """Clears all data from the session."""
    st.session_state.pdf_text = None
    st.session_state.filename = None
    st.session_state.messages = [{"role": "assistant", "content": "Session cleared. Upload a new PDF to start."}]
    log.info("Session cleared.")

# --- Sidebar for Upload and Session Control ---
with st.sidebar:
    st.header("1. Upload PDF")
    uploaded_file = st.file_uploader("Upload your PDF here...", type="pdf")
    
    st.header("2. Session Control")
    st.button("Clear Session & Data", on_click=clear_session_state)
    
    if st.session_state.filename:
        st.info(f"**Active PDF:** {st.session_state.filename}")
    else:
        st.info("No PDF uploaded.")
    
    st.divider()
    st.header("Health Check")
    st.json({
        'status': 'ok',
        'model': MODEL_NAME if genai else ('offline' if ALLOW_OFFLINE else 'not-configured'),
        'pdf2image': bool(convert_from_bytes),
        'pytesseract': bool(pytesseract),
    })

# --- Main App Logic ---

# 1. Handle PDF Upload
if uploaded_file is not None:
    # Check if this is a new file
    if st.session_state.filename != uploaded_file.name:
        log.info(f"New file uploaded: {uploaded_file.name}")
        st.session_state.filename = uploaded_file.name
        
        with st.spinner(f"Processing '{uploaded_file.name}'... This may take a moment."):
            start_time = time.time()
            content = uploaded_file.read()

            # Use your existing extraction logic
            extracted_text, page_images = extract_text_from_pdf(content)
            ocr_text = ''
            if (not extracted_text.strip()) and page_images and pytesseract is not None:
                ocr_text = ocr_pdf_images(page_images)
            
            full_text = (extracted_text + "\n" + ocr_text).strip()
            
            if not full_text:
                st.error("No text could be extracted from this PDF. It might be empty or corrupt.")
                clear_session_state()
            else:
                st.session_state.pdf_text = full_text
                # Reset chat history for the new file
                st.session_state.messages = [
                    {"role": "assistant", 
                     "content": f"Successfully processed **{uploaded_file.name}**."
                                f" (Text length: {len(full_text)} chars). You can now ask questions."}
                ]
                log.info(f"Processed '{uploaded_file.name}'. Elapsed: {time.time() - start_time:.2f}s")
        
        # Rerun to display the new "processed" message immediately
        st.rerun()

# 2. Display Chat History
if not st.session_state.messages and not st.session_state.filename:
     st.info("Please upload a PDF in the sidebar to begin chatting.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Handle New Chat Input
if prompt := st.chat_input("Ask a question about the PDF..."):
    # First, check if a PDF is loaded
    if st.session_state.pdf_text is None:
        st.warning("Please upload a PDF first.", icon="⚠️")
    else:
        # Add user message to state and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                pdf_context = st.session_state.pdf_text
                answer = answer_question_from_context(pdf_context, prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})