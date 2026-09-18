import streamlit as st
import fitz
import ollama
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Research Paper Analyzer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #ffd6e7 0%,
            #ffe6f0 50%,
            #fff0f6 100%
        );
        color: #3f2030;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .stApp p,
    .stApp label {
        color: #3f2030 !important;
    }

    /* =====================================================
       HEADER
       ===================================================== */

    .main-header {
        padding: 32px;
        border-radius: 24px;
        background: rgba(255,255,255,0.70);
        border: 1px solid rgba(185,28,75,0.15);
        backdrop-filter: blur(15px);
        margin-bottom: 28px;
        box-shadow: 0 15px 40px rgba(120,20,60,0.12);
    }

    .main-header h1 {
        color: #8f1742 !important;
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .main-header p {
        color: #5a3042 !important;
        font-size: 17px;
    }

    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #b91c4b 0%,
            #d6336c 55%,
            #e85d8a 100%
        );
        border-right: 1px solid rgba(255,255,255,0.25);
    }

   section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: #0000ff !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
    color: #0000ff !important;
}

    /* =====================================================
       ZIA PROFILE
       ===================================================== */

    .profile-card {
        text-align: center;
        padding: 20px 5px 20px 5px;
    }

    .profile-icon {
        font-size: 65px;
        margin-bottom: 12px;
    }

    .profile-name {
        color: white !important;
        font-size: 23px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .profile-role {
        color: white !important;
        font-size: 14px;
        line-height: 1.7;
    }

    /* =====================================================
       UPLOAD
       ===================================================== */

    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.70);
        padding: 18px;
        border-radius: 18px;
        border: 2px dashed rgba(185,28,75,0.25);
    }

    [data-testid="stFileUploader"] section {
        background: transparent !important;
    }

    /* =====================================================
       QUESTION
       ===================================================== */

    .stTextInput input {
        color: #000000 !important;
        background-color: #ffffff !important;
        border: 2px solid #d6336c !important;
        border-radius: 14px !important;
        padding: 14px !important;
        font-size: 16px !important;
        caret-color: #000000 !important;
    }

    .stTextInput input::placeholder {
        color: #666666 !important;
        opacity: 1 !important;
    }

    .stTextInput input:focus {
        border-color: #b91c4b !important;
        box-shadow: 0 0 0 2px rgba(185,28,75,0.12) !important;
    }

    /* =====================================================
       ANSWER
       ===================================================== */

    .answer-card {
        padding: 28px;
        border-radius: 20px;
        background: rgba(255,255,255,0.82);
        border: 1px solid rgba(185,28,75,0.15);
        box-shadow: 0 15px 40px rgba(120,20,60,0.12);
        margin-top: 10px;
    }

    .answer-card p,
    .answer-card li,
    .answer-card strong {
        color: #3f2030 !important;
        line-height: 1.75;
    }

    /* =====================================================
       SOURCE
       ===================================================== */

    .source-heading {
        color: #9f1748 !important;
        font-size: 17px;
        font-weight: 700;
    }

    /* =====================================================
       GENERAL
       ===================================================== */

    h1, h2, h3, h4 {
        color: #8f1742 !important;
    }

    .streamlit-expanderHeader {
        color: #8f1742 !important;
        background: rgba(255,255,255,0.60) !important;
        border-radius: 12px;
    }

    hr {
        border-color: rgba(185,28,75,0.15) !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    

    st.markdown("### 🧠 About")

    st.write(
        """
        AI-powered research paper analysis using
        local embeddings, FAISS semantic search,
        Retrieval-Augmented Generation (RAG),
        and Llama AI.
        """
    )

    st.divider()

    st.markdown("### ⚙️ Technology")

    st.write(
        """
        🐍 Python  
        🎈 Streamlit  
        🧠 Sentence Transformers  
        🔎 FAISS  
        🦙 Ollama / Llama 3.2  
        📄 PyMuPDF
        """
    )

    st.divider()

    st.markdown("### 👨‍💻 Developer")

    st.write(
        """
        **Engr. Muhammad Zia**

        Software Engineering  
        Artificial Intelligence  
        Data Science
        """
    )

    st.divider()

    st.caption(
        "Developed by Engr. Muhammad Zia"
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="main-header">

        <h1>📚 AI Research Paper Analyzer</h1>

        <p>
            Understand research papers faster with
            local AI-powered semantic search and
            intelligent question answering.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# UPLOAD
# =========================================================

st.markdown("### 📤 Upload Research Paper")

uploaded_file = st.file_uploader(
    "Choose your PDF research paper",
    type=["pdf"]
)


# =========================================================
# EMBEDDINGS
# =========================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_pages_from_pdf(pdf_file):

    pdf_bytes = pdf_file.getvalue()

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(
        document,
        start=1
    ):

        text = page.get_text("text")

        if text:

            text = re.sub(
                r"\s+",
                " ",
                text
            ).strip()

            if len(text) > 50:

                pages.append(
                    Document(
                        page_content=text,
                        metadata={
                            "page": page_number
                        }
                    )
                )

    document.close()

    return pages


# =========================================================
# CHUNKING
# =========================================================

def create_chunks(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",
            ". ",
            "\n",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(
        documents
    )

    clean_chunks = []

    for chunk in chunks:

        text = chunk.page_content.strip()

        if len(text) >= 100:

            clean_chunks.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": chunk.metadata.get(
                            "page",
                            "Unknown"
                        )
                    }
                )
            )

    return clean_chunks


# =========================================================
# VECTOR STORE
# =========================================================

@st.cache_resource
def create_vector_store(
    chunk_texts,
    chunk_pages
):

    embeddings = load_embeddings()

    documents = []

    for text, page in zip(
        chunk_texts,
        chunk_pages
    ):

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "page": page
                }
            )
        )

    return FAISS.from_documents(
        documents,
        embedding=embeddings
    )


# =========================================================
# QUESTION TYPE
# =========================================================

def is_overview_question(question):

    q = question.lower().strip()

    overview_patterns = [
        "what is this paper about",
        "what is the paper about",
        "what does this paper discuss",
        "what is this research about",
        "what is the main idea",
        "what's the main idea",
        "main idea of this paper",
        "main topic of this paper",
        "what is the main topic",
        "summarize this paper",
        "summarise this paper",
        "give me a summary",
        "paper summary",
        "summary of this paper",
        "what does this paper propose",
        "what is the purpose of this paper",
        "what are the main contributions",
        "main contributions of this paper",
        "explain this paper",
        "describe this paper",
        "give me an overview"
    ]

    return any(
        pattern in q
        for pattern in overview_patterns
    )


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_relevant_chunks(
    vectorstore,
    question,
    all_chunks,
    k=5
):

    if is_overview_question(question):

        early_chunks = [
            chunk
            for chunk in all_chunks
            if isinstance(
                chunk.metadata.get("page"),
                int
            )
            and chunk.metadata.get("page") <= 2
        ]

        if early_chunks:

            early_chunks = sorted(
                early_chunks,
                key=lambda x: (
                    x.metadata.get("page", 999),
                    all_chunks.index(x)
                )
            )

            return early_chunks[:k]

    results = vectorstore.similarity_search(
        question,
        k=max(k, 8)
    )

    unique_results = []
    seen = set()

    for result in results:

        text_key = result.page_content.strip()

        if text_key not in seen:

            seen.add(text_key)
            unique_results.append(result)

        if len(unique_results) >= k:
            break

    return unique_results


# =========================================================
# LLAMA
# =========================================================

def ask_llama(
    question,
    context
):

    prompt = f"""
You are an AI research paper analysis assistant.

The user uploaded a research paper and asked a question.

Your answer MUST be based ONLY on the provided paper
context below.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent information.
3. Ignore references, bibliography entries, captions,
   examples, or unrelated sentences if they do not answer
   the user's question.
4. Identify the actual research topic from the paper.
5. For questions about the paper's main idea, purpose,
   contribution, or summary, focus on the title, abstract,
   introduction, and central research contribution.
6. Do not mistake an example inside the paper for the
   main topic of the paper.
7. Answer clearly and directly.
8. Mention the relevant page number when supported by
   the context.
9. If the answer cannot be supported by the context, say:
   "The answer was not found in the provided paper sections."

RESEARCH PAPER CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# =========================================================
# APPLICATION
# =========================================================

if uploaded_file:

    # =====================================================
    # EXTRACT
    # =====================================================

    with st.spinner(
        "📖 Reading research paper..."
    ):

        pages = extract_pages_from_pdf(
            uploaded_file
        )

    if not pages:

        st.error(
            "❌ No readable text was found in this PDF."
        )

        st.stop()

    st.success(
        f"✅ Uploaded: {uploaded_file.name}"
    )


    # =====================================================
    # CHUNKS
    # =====================================================

    with st.spinner(
        "✂️ Creating searchable sections..."
    ):

        chunks = create_chunks(
            pages
        )

    if not chunks:

        st.error(
            "❌ Could not create searchable sections."
        )

        st.stop()


    # =====================================================
    # VECTOR DATABASE
    # =====================================================

    with st.spinner(
        "🧠 Building semantic search database..."
    ):

        chunk_texts = [
            chunk.page_content
            for chunk in chunks
        ]

        chunk_pages = [
            chunk.metadata.get(
                "page",
                "Unknown"
            )
            for chunk in chunks
        ]

        vectorstore = create_vector_store(
            tuple(chunk_texts),
            tuple(chunk_pages)
        )

    st.success(
        "✅ FAISS vector database is ready!"
    )


    # =====================================================
    # QUESTION
    # =====================================================

    st.markdown(
        "### 🤖 Ask Your Research Paper"
    )

    question = st.text_input(
        "Ask anything about the paper",
        placeholder="Example: What is this paper about?",
        key="research_question"
    )


    # =====================================================
    # PROCESS QUESTION
    # =====================================================

    if question.strip():

        with st.spinner(
            "🔎 Searching relevant sections..."
        ):

            results = retrieve_relevant_chunks(
                vectorstore,
                question,
                chunks,
                k=5
            )

        context_parts = []

        for result in results:

            page_number = result.metadata.get(
                "page",
                "Unknown"
            )

            context_parts.append(
                f"""
[Page {page_number}]

{result.page_content}
"""
            )

        context = "\n\n".join(
            context_parts
        )

        with st.spinner(
            "🦙 Llama is analyzing the paper..."
        ):

            try:

                answer = ask_llama(
                    question,
                    context
                )

                st.markdown(
                    "### 💡 AI Answer"
                )

                st.markdown(
                    '<div class="answer-card">',
                    unsafe_allow_html=True
                )

                st.markdown(answer)

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

            except Exception as e:

                st.error(
                    f"""
                    ❌ Ollama connection error.

                    Make sure Ollama is running and
                    llama3.2:3b is installed.

                    Error:
                    {e}
                    """
                )

        with st.expander(
            "📚 View Retrieved Sources"
        ):

            if results:

                for i, result in enumerate(
                    results,
                    start=1
                ):

                    page_number = result.metadata.get(
                        "page",
                        "Unknown"
                    )

                    st.markdown(
                        f"**📄 Source {i} — Page {page_number}**"
                    )

                    st.write(
                        result.page_content
                    )

                    st.divider()

            else:

                st.info(
                    "No relevant sources were found."
                )


# =========================================================
# EMPTY STATE
# =========================================================

else:

    st.info(
        "📄 Please upload a research paper PDF to start."
    )

    st.markdown(
        """
        <div class="answer-card">

            <h3>👨‍💻 Engr. Muhammad Zia</h3>

            <p>
                Welcome to the AI Research Paper Analyzer.
                Upload a research paper and ask questions
                about its content using local AI.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )