import streamlit as st
import fitz
import re
from dotenv import load_dotenv
import os
from datetime import datetime

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains import RetrievalQA

try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except:
    from dotenv import load_dotenv
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# ======================================================
# ---------------  TEXT CLEANING -----------------------
# ======================================================
def clean_text(raw_text):
    return re.sub(r"[^\w\s.,;:!?()-]", "", raw_text)

# ======================================================
# ---------------  PDF EXTRACTION ----------------------
# ======================================================
def extract_text(pdf_file):
    pdf_doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in pdf_doc:
        text += page.get_text("text")
    pdf_doc.close()
    return clean_text(text)

def get_pdf_info(pdf_file):
    """Extract PDF metadata"""
    pdf_doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    info = {
        "pages": len(pdf_doc),
        "title": pdf_doc.metadata.get("title", "Unknown"),
    }
    pdf_doc.close()
    return info

# ======================================================
# ---------------  FAISS VECTOR DB ---------------------
# ======================================================
def create_faiss_db(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=c) for c in chunks]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.from_documents(docs, embeddings)
    return db, len(chunks)

# ======================================================
# ---------------  QA CHAIN ----------------------------
# ======================================================
def create_qa_chain(db):
    retriever = db.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt_template = """
You are an expert assistant for NIT Uttarakhand students.
Use ONLY the provided context to answer.
If answer not found, reply: "I don't have that information in the document. Please ask something else!"

Context:
{context}

Question:
{question}

Answer:
"""
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

# ======================================================
# ---------------  STYLING -----------------------------
# ======================================================
st.set_page_config(page_title="NIT UK PDF Assistant", layout="wide", page_icon="🎓")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.main {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
    background-attachment: fixed;
}

.block-container {
    background: rgba(255, 255, 255, 0.98);
    border-radius: 30px;
    padding: 2.5rem;
    box-shadow: 0 25px 80px rgba(0,0,0,0.4);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.3);
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%) !important;
}

section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%) !important;
    padding: 2rem 1rem;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

section[data-testid="stSidebar"] .stMarkdown {
    color: white !important;
}

section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
    color: white !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

section[data-testid="stSidebar"] .stRadio > label {
    background: rgba(255, 255, 255, 0.15) !important;
    padding: 1rem;
    border-radius: 15px;
    margin: 0.5rem 0;
    backdrop-filter: blur(5px);
    transition: all 0.3s ease;
    border: 2px solid rgba(255, 255, 255, 0.2);
}

section[data-testid="stSidebar"] .stRadio > label:hover {
    background: rgba(255, 255, 255, 0.25) !important;
    transform: translateX(8px);
    border-color: rgba(255, 255, 255, 0.4);
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
}

/* Sidebar Info/Success/Warning boxes */
section[data-testid="stSidebar"] .stAlert {
    background: rgba(255, 255, 255, 0.2) !important;
    border-radius: 12px !important;
    border-left: 4px solid rgba(255, 255, 255, 0.6) !important;
    backdrop-filter: blur(10px);
    color: white !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] .stAlert div {
    color: white !important;
}

/* Logo Container in Sidebar */
.logo-container {
    text-align: center;
    padding: 1.5rem;
    background: rgba(255, 255, 255, 0.15);
    border-radius: 20px;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
    border: 2px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
}

.logo-container img {
    border-radius: 50%;
    border: 4px solid rgba(255, 255, 255, 0.3);
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    transition: transform 0.3s ease;
}

.logo-container img:hover {
    transform: scale(1.05) rotate(5deg);
}

.logo-text {
    color: white !important;
    font-weight: 700;
    margin-top: 1rem;
    font-size: 1.1rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

/* Button Styling */
.stButton>button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 30px;
    padding: 0.85rem 2.5rem;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.5px;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    text-transform: uppercase;
}

.stButton>button:hover {
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0 15px 40px rgba(102, 126, 234, 0.7);
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}

.stButton>button:active {
    transform: translateY(-2px);
}

/* Chat Bubbles with Glass Effect */
.chat-bubble-user {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%);
    color: white;
    padding: 1.25rem 1.75rem;
    border-radius: 25px 25px 8px 25px;
    margin: 0.75rem 0;
    margin-left: 15%;
    animation: slideInRight 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    font-weight: 500;
}

.chat-bubble-bot {
    background: linear-gradient(135deg, rgba(240, 147, 251, 0.95) 0%, rgba(245, 87, 108, 0.95) 100%);
    color: white;
    padding: 1.25rem 1.75rem;
    border-radius: 25px 25px 25px 8px;
    margin: 0.75rem 0;
    margin-right: 15%;
    animation: slideInLeft 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 10px 30px rgba(245, 87, 108, 0.4);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    font-weight: 500;
}

@keyframes slideInRight {
    from {
        opacity: 0;
        transform: translateX(80px) scale(0.9);
    }
    to {
        opacity: 1;
        transform: translateX(0) scale(1);
    }
}

@keyframes slideInLeft {
    from {
        opacity: 0;
        transform: translateX(-80px) scale(0.9);
    }
    to {
        opacity: 1;
        transform: translateX(0) scale(1);
    }
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-15px); }
}

@keyframes glow {
    0%, 100% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.5); }
    50% { box-shadow: 0 0 40px rgba(102, 126, 234, 0.8); }
}

/* Stat Cards with Hover Effects */
.stat-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem 1.5rem;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 10px 35px rgba(102, 126, 234, 0.4);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    border: 2px solid rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
}

.stat-card::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
    transform: rotate(45deg);
    transition: all 0.5s ease;
}

.stat-card:hover::before {
    left: 100%;
}

.stat-card:hover {
    transform: translateY(-10px) scale(1.03);
    box-shadow: 0 20px 50px rgba(102, 126, 234, 0.6);
    animation: glow 2s infinite;
}

.stat-card h2 {
    font-size: 2.5rem;
    font-weight: 800;
    margin: 0.5rem 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}

/* Header Title with Animation */
.header-title {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 3.5rem;
    font-weight: 900;
    text-align: center;
    margin-bottom: 0;
    animation: float 3s ease-in-out infinite;
    letter-spacing: 1px;
}

.subtitle {
    text-align: center;
    color: #555;
    font-size: 1.3rem;
    margin-top: 0.5rem;
    font-weight: 500;
    letter-spacing: 0.5px;
}

/* Input Field Styling */
.stTextInput>div>div>input {
    border-radius: 30px;
    border: 3px solid #667eea;
    padding: 1rem 2rem;
    font-size: 1.1rem;
    transition: all 0.3s ease;
    background: rgba(255, 255, 255, 0.9);
    font-weight: 500;
}

.stTextInput>div>div>input:focus {
    border-color: #764ba2;
    box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.3);
    transform: scale(1.02);
}

/* Feature Badges */
.feature-badge {
    display: inline-block;
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 0.75rem 1.5rem;
    border-radius: 25px;
    margin: 0.5rem;
    font-weight: 700;
    font-size: 1rem;
    box-shadow: 0 5px 15px rgba(245, 87, 108, 0.4);
    transition: all 0.3s ease;
    border: 2px solid rgba(255, 255, 255, 0.3);
}

.feature-badge:hover {
    transform: translateY(-5px) scale(1.05);
    box-shadow: 0 10px 25px rgba(245, 87, 108, 0.6);
}

/* Steps Cards */
.step-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    border: 2px solid rgba(255, 255, 255, 0.2);
    position: relative;
    overflow: hidden;
}

.step-card::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1));
    transform: translateX(-100%);
    transition: transform 0.6s ease;
}

.step-card:hover::after {
    transform: translateX(100%);
}

.step-card:hover {
    transform: translateY(-10px) rotate(2deg);
    box-shadow: 0 20px 50px rgba(102, 126, 234, 0.6);
}

.step-number {
    font-size: 3rem;
    font-weight: 900;
    opacity: 0.9;
    text-shadow: 3px 3px 6px rgba(0,0,0,0.2);
}

/* Info boxes */
.stInfo, .stSuccess, .stWarning {
    border-radius: 15px;
    border-left: 5px solid;
    font-weight: 500;
}

/* ========= ENHANCED FILE UPLOADER STYLING ========= */
section[data-testid="stFileUploader"] {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.3), rgba(255, 255, 255, 0.2)) !important;
    border-radius: 25px !important;
    padding: 2.5rem !important;
    border: 4px dashed rgba(255, 215, 0, 0.8) !important;
    backdrop-filter: blur(15px) !important;
    transition: all 0.4s ease !important;
    box-shadow: 0 15px 50px rgba(0, 0, 0, 0.4) !important;
    margin: 1rem 0 !important;
}

section[data-testid="stFileUploader"]:hover {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.4), rgba(255, 255, 255, 0.3)) !important;
    border-color: rgba(255, 215, 0, 1) !important;
    transform: scale(1.03) !important;
    box-shadow: 0 20px 60px rgba(255, 215, 0, 0.3) !important;
}

section[data-testid="stFileUploader"] label {
    color: #FFD700 !important;
    font-weight: 900 !important;
    font-size: 1.3rem !important;
    text-shadow: 3px 3px 8px rgba(0,0,0,0.6) !important;
    margin-bottom: 1.5rem !important;
    display: block !important;
    letter-spacing: 1px !important;
}

section[data-testid="stFileUploader"] small {
    color: rgba(255, 255, 255, 1) !important;
    font-weight: 700 !important;
    text-shadow: 2px 2px 5px rgba(0,0,0,0.5) !important;
    font-size: 1rem !important;
    background: rgba(0, 0, 0, 0.3) !important;
    padding: 0.3rem 0.8rem !important;
    border-radius: 10px !important;
    display: inline-block !important;
}

/* File uploader drag and drop area */
section[data-testid="stFileUploader"] > div > div {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.25), rgba(255, 255, 255, 0.15)) !important;
    border: 4px dashed rgba(255, 255, 255, 0.8) !important;
    border-radius: 20px !important;
    padding: 3.5rem 2rem !important;
    min-height: 200px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.3s ease !important;
}

section[data-testid="stFileUploader"] > div > div:hover {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.35), rgba(255, 255, 255, 0.25)) !important;
    border-color: rgba(255, 255, 255, 1) !important;
    border-width: 5px !important;
}

/* Browse files button - SUPER VISIBLE */
section[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 50%, #FFD93D 100%) !important;
    color: #1a1a2e !important;
    border: 3px solid rgba(255, 255, 255, 0.9) !important;
    border-radius: 25px !important;
    padding: 1.2rem 3rem !important;
    font-weight: 900 !important;
    font-size: 1.2rem !important;
    box-shadow: 0 10px 40px rgba(255, 107, 107, 0.6) !important;
    transition: all 0.4s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.5) !important;
}

section[data-testid="stFileUploader"] button:hover {
    transform: translateY(-8px) scale(1.08) !important;
    box-shadow: 0 15px 50px rgba(255, 107, 107, 0.8) !important;
    background: linear-gradient(135deg, #FFD93D 0%, #FF8E53 50%, #FF6B6B 100%) !important;
    border-color: rgba(255, 255, 255, 1) !important;
}

/* File upload text and icons */
section[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {
    color: rgba(255, 255, 255, 1) !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    text-shadow: 2px 2px 5px rgba(0,0,0,0.5) !important;
    background: rgba(0, 0, 0, 0.2) !important;
    padding: 0.5rem 1rem !important;
    border-radius: 10px !important;
    display: inline-block !important;
}

/* Drag and drop text */
section[data-testid="stFileUploader"] span {
    color: rgba(255, 255, 255, 1) !important;
    font-weight: 700 !important;
    text-shadow: 2px 2px 5px rgba(0,0,0,0.5) !important;
}

/* Uploaded file display */
section[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    background: linear-gradient(135deg, rgba(255, 215, 0, 0.3), rgba(255, 255, 255, 0.3)) !important;
    border-radius: 15px !important;
    padding: 1.2rem !important;
    border: 3px solid rgba(255, 215, 0, 0.6) !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 0 5px 20px rgba(255, 215, 0, 0.3) !important;
}

section[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button {
    background: linear-gradient(135deg, #FF6B6B, #FF8E53) !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.2rem !important;
    border: 2px solid white !important;
}

/* File name text */
section[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] {
    color: white !important;
    font-weight: 700 !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5) !important;
    font-size: 1.05rem !important;
}

/* Divider */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.5), transparent);
    margin: 2rem 0;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# --------------- SIDEBAR ------------------------------
# ======================================================
with st.sidebar:
    # Logo with enhanced styling
    # Line ~547 ke around
    logo_path = "logo.jpeg"
    if os.path.exists(logo_path):
        st.markdown("""
        <div class='logo-container'>
        """, unsafe_allow_html=True)
        st.image(logo_path, width=180)
        st.markdown("""
            <p class='logo-text'>NIT UTTARAKHAND</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='logo-container'>
            <p class='logo-text'>NIT UTTARAKHAND</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📚 Document Control Center")
    
    uploaded_file = st.file_uploader("Upload Your PDF", type=["pdf"], help="Max 200MB • PDF Format Only")
    
    st.markdown("---")
    
    menu = st.radio(
        "🧭 Navigation",
        ["💬 Chat", "📊 Analytics", "📄 Preview", "📝 History", "ℹ️ About"],
        index=0
    )
    
    st.markdown("---")
    
    if uploaded_file:
        if st.button("🔄 New Session", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 🎯 Quick Tips")
    st.info("💡 Ask specific questions for better results!")
    st.success("✨ Use simple, clear language")
    st.warning("⚡ Be patient with large PDFs")

# ======================================================
# ---------------  HEADER + LOGO -----------------------
# ======================================================
st.markdown("<h1 class='header-title'>🎓 NIT UK AI Document Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>✨ Your Intelligent PDF Companion • Powered by Advanced AI</p>", unsafe_allow_html=True)

st.markdown("---")

# ======================================================
# --------------- MAIN LOGIC ---------------------------
# ======================================================
if not uploaded_file:
    st.markdown("### 🚀 Get Started in 3 Simple Steps")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='step-card'>
            <div class='step-number'>1</div>
            <h3>📤 Upload</h3>
            <p>Choose your PDF document</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='step-card'>
            <div class='step-number'>2</div>
            <h3>❓ Ask</h3>
            <p>Type your questions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='step-card'>
            <div class='step-number'>3</div>
            <h3>✨ Get Answers</h3>
            <p>Instant AI responses</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### ✨ Premium Features")
    st.markdown("""
    <div style='text-align: center;'>
        <span class='feature-badge'>🔍 Smart Search</span>
        <span class='feature-badge'>⚡ Instant Answers</span>
        <span class='feature-badge'>📊 Analytics Dashboard</span>
        <span class='feature-badge'>💾 Chat History</span>
        <span class='feature-badge'>🎨 Beautiful Interface</span>
        <span class='feature-badge'>🔒 Secure Processing</span>
        <span class='feature-badge'>🚀 Lightning Fast</span>
        <span class='feature-badge'>📱 Responsive Design</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Upload a PDF from the sidebar to unlock AI-powered insights!")
    st.stop()

# Process PDF
if "processed" not in st.session_state:
    with st.spinner("🔄 Processing your PDF with AI... Please wait"):
        uploaded_file.seek(0)
        text = extract_text(uploaded_file)
        uploaded_file.seek(0)
        pdf_info = get_pdf_info(uploaded_file)
        uploaded_file.seek(0)
        db, num_chunks = create_faiss_db(text)
        qa_chain = create_qa_chain(db)
        pdf_bytes = uploaded_file.getvalue()
        
        st.session_state.processed = True
        st.session_state.text = text
        st.session_state.db = db
        st.session_state.qa_chain = qa_chain
        st.session_state.pdf_bytes = pdf_bytes
        st.session_state.pdf_info = pdf_info
        st.session_state.num_chunks = num_chunks
        st.balloons()
        st.success("✅ PDF processed successfully! Start asking questions.")

# Session state for chat
if "chat" not in st.session_state:
    st.session_state.chat = []

# ======================================================
# --------------- MENU HANDLING ------------------------
# ======================================================

# CHAT MODE
if menu == "💬 Chat":
    st.markdown("### 💬 Intelligent Q&A System")
    
    # Quick question buttons
    st.markdown("#### 🎯 Suggested Questions")
    col1, col2, col3 = st.columns(3)
    
    quick_questions = [
        "What is this document about?",
        "Summarize the key points",
        "What are the main topics?"
    ]
    
    selected_quick = None
    with col1:
        if st.button("📝 " + quick_questions[0], use_container_width=True):
            selected_quick = quick_questions[0]
    with col2:
        if st.button("📋 " + quick_questions[1], use_container_width=True):
            selected_quick = quick_questions[1]
    with col3:
        if st.button("📚 " + quick_questions[2], use_container_width=True):
            selected_quick = quick_questions[2]
    
    st.markdown("---")
    
    question = st.text_input("✍️ Type your question here:", value=selected_quick if selected_quick else "", key="question_input", placeholder="Ask me anything about your PDF...")
    
    if question:
        with st.spinner("🤔 AI is thinking..."):
            result = st.session_state.qa_chain.invoke({"query": question})
            answer = result["result"]
            timestamp = datetime.now().strftime("%H:%M")
            st.session_state.chat.append(("user", question, timestamp))
            st.session_state.chat.append(("bot", answer, timestamp))

    # Chat display with timestamps
    if st.session_state.chat:
        st.markdown("### 💭 Conversation Thread")
        for sender, msg, timestamp in st.session_state.chat:
            if sender == "user":
                st.markdown(f"<div class='chat-bubble-user'><b>👤 You</b> <small style='opacity: 0.8;'>• {timestamp}</small><br><br>{msg}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-bubble-bot'><b>🤖 AI Assistant</b> <small style='opacity: 0.8;'>• {timestamp}</small><br><br>{msg}</div>", unsafe_allow_html=True)
    else:
        st.info("👋 Start the conversation by asking a question above!")

# ANALYTICS MODE
elif menu == "📊 Analytics":
    st.markdown("### 📊 Document Analytics Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='stat-card'>
            <h2>{st.session_state.pdf_info['pages']}</h2>
            <p style='font-size: 1.1rem; font-weight: 600;'>📄 Total Pages</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='stat-card'>
            <h2>{st.session_state.num_chunks}</h2>
            <p style='font-size: 1.1rem; font-weight: 600;'>🧩 Text Chunks</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='stat-card'>
            <h2>{len(st.session_state.chat) // 2}</h2>
            <p style='font-size: 1.1rem; font-weight: 600;'>❓ Questions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        word_count = len(st.session_state.text.split())
        st.markdown(f"""
        <div class='stat-card'>
            <h2>{word_count:,}</h2>
            <p style='font-size: 1.1rem; font-weight: 600;'>📝 Words</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📄 Document Preview")
    st.text_area("First 500 characters of your document:", st.session_state.text[:500] + "...", height=250)

# PDF PREVIEW
elif menu == "📄 Preview":
    st.markdown("### 📄 PDF Document Viewer")
    st.download_button("⬇️ Download Original PDF", data=st.session_state.pdf_bytes, file_name="document.pdf", use_container_width=True)
    st.markdown("---")
    st.info("📖 Viewing your uploaded PDF document")

# CHAT HISTORY
elif menu == "📝 History":
    st.markdown("### 📝 Complete Conversation History")
    if not st.session_state.chat:
        st.info("🤷 No conversations yet! Start asking questions in the Chat section.")
    else:
        for s, m, t in st.session_state.chat:
            if s == "user":
                st.markdown(f"**🙋 You ({t}):** {m}")
            else:
                st.markdown(f"**🤖 AI Assistant ({t}):** {m}")
            st.markdown("---")
        
        chat_text = "\n\n".join([f"{s.upper()} ({t}): {m}" for s, m, t in st.session_state.chat])
        st.download_button(
            "📥 Download Complete Chat History",
            data=chat_text,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            use_container_width=True
        )

# ABOUT
elif menu == "ℹ️ About":
    st.markdown("### ℹ️ About This Application")
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 3rem; border-radius: 25px; color: white; box-shadow: 0 20px 60px rgba(102, 126, 234, 0.5);'>
        <h1 style='margin: 0; font-size: 2.5rem;'>🎓 NIT Uttarakhand PDF Assistant</h1>
        <p style='font-size: 1.3rem; margin-top: 1rem; opacity: 0.95;'>An intelligent document analysis system powered by cutting-edge AI technology</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🛠️ Tech Stack")
        st.markdown("""
        - **Framework:** Streamlit
        - **AI/ML:** LangChain + OpenAI GPT-4
        - **Vector Database:** FAISS
        - **PDF Processing:** PyMuPDF (fitz)
        - **Embeddings:** OpenAI text-embedding-3-small
        """)
    
    with col2:
        st.markdown("### 👨‍💻 Developer")
        st.markdown("""
        **Gautam Singh (gautamkumarsing01@gmail.com)**
        - 🎓 NIT Uttarakhand
        - 💻 Full Stack Developer
        - 🤖 AI/ML Enthusiast
        - 📧 Open for collaboration
        - ⭐ Made with ❤️ in India
        """)
    
    st.markdown("---")
    st.success("💡 **Pro Tip:** For best results, ask specific and clear questions about your document content!")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem; font-weight: 500;'>© 2025 | Crafted with ❤️ by Gautam Singh | NIT Uttarakhand 🎓</p>", unsafe_allow_html=True)