import streamlit as st
import fitz  # PyMuPDF
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests
import re
import nltk
from nltk.tokenize import sent_tokenize

# --- Page Configuration MUST BE FIRST ---
st.set_page_config(page_title="LegalDoc AI Summarizer + NLP", page_icon="⚖️", layout="wide")

# --- Initialization ---
# Ensure NLTK data is available for sentence tokenization
@st.cache_resource
def download_nltk_data():
    try:
        # Check if punkt is already downloaded
        nltk.data.find('tokenizers/punkt')
        # Also check for the newer punkt_tab if using latest nltk
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
    except LookupError:
        with st.spinner("Downloading NLP components (one-time setup)..."):
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        st.warning(f"Note: NLP sentence segmentation might be limited: {e}")

download_nltk_data()

st.title("⚖️ Local Legal Document Summarizer (NLP Preprocessing)")
st.markdown("""
    Summarize complex legal contracts locally with **Ollama** and **NLP-enhanced extraction**.
    *Cleans PDF artifacts and normalizes text for better LLM performance.*
""")

# --- Sidebar ---
st.sidebar.header("Configuration")
model_name = st.sidebar.selectbox("Select Ollama Model", ["llama3", "mistral", "llama2"], index=0)
st.sidebar.divider()
uploaded_file = st.sidebar.file_uploader("Upload Legal PDF", type=["pdf"])

# --- NLP Preprocessing Functions ---
def preprocess_legal_text(text):
    """
    Cleans raw text from PDF to improve LLM comprehension.
    1. Normalizes whitespace
    2. Fixes end-of-line hyphens (e.g., 'agree- \nment' -> 'agreement')
    3. Standardizes common legal characters
    4. Segments by sentences to ensure clean chunking
    """
    # 1. Fix end-of-line hyphens
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # 2. Normalize whitespace (tabs, multiple spaces, etc.)
    text = re.sub(r'\s+', ' ', text)
    
    # 3. Standardize quotes and special characters
    text = text.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'")
    
    # 4. Remove common PDF extraction noise (like multiple dots)
    text = re.sub(r'\.{3,}', '...', text)
    
    # 5. Segment into sentences (NLTK) to re-verify structure
    try:
        sentences = sent_tokenize(text)
        text = " ".join(sentences)
    except:
        pass # Fallback to original text if sentence tokenization fails
        
    return text.strip()

def extract_text_from_pdf(file):
    """Extracts text from a PDF file using PyMuPDF."""
    try:
        file_bytes = file.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def check_ollama_connection():
    """Checks if the local Ollama server is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

# --- Main Logic ---
if uploaded_file:
    if not check_ollama_connection():
        st.error("❌ **Ollama server not detected.**")
        st.info("Run `ollama serve` and ensure you have pulled the model (`ollama pull llama3`).")
    else:
        with st.status("Analyzing Document...", expanded=True) as status:
            # 1. Extraction
            st.write("📥 Extracting PDF content...")
            raw_text = extract_text_from_pdf(uploaded_file)
            
            if raw_text and len(raw_text.strip()) > 100:
                # 2. NLP Preprocessing
                st.write("⚙️ Applying NLP preprocessing...")
                cleaned_text = preprocess_legal_text(raw_text)
                
                # 3. Chunking
                st.write("📑 Chunking cleaned text...")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000, 
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ".", " "]
                )
                chunks = text_splitter.split_text(cleaned_text)
                
                # 4. Define Map-Reduce Prompts
                map_template = """
                Extract key legal entities, obligations, and terms from the following text:
                "{text}"
                CONCISE SUMMARY OF THIS SECTION:
                """
                map_prompt = PromptTemplate.from_template(map_template)

                reduce_template = """
                The following are summaries from a legal document:
                "{text}"
                
                Provide a cohesive final report with:
                1. **Core Summary**: High-level essence of the document.
                2. **Key Obligations & Terms**: Bulleted list of duties and conditions.
                3. **Potential Risks / Red Flags**: Bulleted list of ambiguous or one-sided terms.
                
                FINAL LEGAL SUMMARY:
                """
                reduce_prompt = PromptTemplate.from_template(reduce_template)

                # 5. Initialize LLM & Pipeline
                llm = OllamaLLM(model=model_name)
                map_chain = map_prompt | llm | StrOutputParser()
                reduce_chain = reduce_prompt | llm | StrOutputParser()
                
                # MAP STEP
                summaries = []
                progress_bar = st.progress(0)
                for i, chunk in enumerate(chunks):
                    st.write(f"Processing segment {i+1} of {len(chunks)}...")
                    summary = map_chain.invoke({"text": chunk})
                    summaries.append(summary)
                    progress_bar.progress((i + 1) / len(chunks))
                
                # REDUCE STEP
                st.write("📊 Synthesizing final report...")
                combined_summaries = "\n\n".join(summaries)
                final_summary = reduce_chain.invoke({"text": combined_summaries})

                status.update(label="Summarization Complete!", state="complete", expanded=False)
                
                # --- Display Results ---
                st.divider()
                st.subheader(f"📄 Analysis: {uploaded_file.name}")
                
                # Show cleaning stats
                with st.expander("Show NLP Stats"):
                    st.write(f"Original Characters: {len(raw_text)}")
                    st.write(f"Preprocessed Characters: {len(cleaned_text)}")
                    st.write(f"Reduction through normalization: {((len(raw_text)-len(cleaned_text))/len(raw_text))*100:.2f}%")
                
                st.markdown(final_summary)
            else:
                st.warning("The PDF is empty or has too little text.")
else:
    st.info("👈 Upload a legal document in the sidebar to begin.")
    st.markdown("---")
    st.image("https://images.unsplash.com/photo-1589829545856-d10d557cf95f?auto=format&fit=crop&q=80&w=1000", caption="Local Legal AI", use_column_width=True)
