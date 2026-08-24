import hashlib
import io
import os
import shutil
import tempfile

import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from utils import ollama_error

st.title("PDF Q&A with Ollama")

PROMPT = PromptTemplate(
    template=(
        "You are an assistant answering questions about a document the user uploaded.\n"
        "Use the following excerpts from that document to answer.\n"
        "- If the question asks what the document is about, or asks for a summary, "
        "describe its overall subject based on the excerpts.\n"
        "- Otherwise answer the question using the excerpts.\n"
        "- Only say you don't know if the excerpts clearly do not contain the answer.\n\n"
        "Document excerpts:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
    input_variables=["context", "question"],
)


def load_pdf(path):
    """Extract text with pdfplumber; fall back to OCR for scanned PDFs."""
    docs = PDFPlumberLoader(path).load()
    if any(d.page_content.strip() for d in docs):
        return docs

    # No extractable text -> image-based PDF, run OCR
    if not shutil.which("tesseract"):
        raise ValueError(
            "This PDF has no text layer (it looks scanned) and tesseract OCR "
            "is not installed. Install it (see README) or use a text-based PDF."
        )

    import fitz  # pymupdf
    import pytesseract
    from PIL import Image

    st.info("No text layer found – running OCR (this can take a minute)...")
    ocr_docs = []
    pdf = fitz.open(path)
    for i, page in enumerate(pdf):
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        txt = pytesseract.image_to_string(img)
        if txt.strip():
            ocr_docs.append(Document(page_content=txt, metadata={"page": i}))
    return ocr_docs


@st.cache_resource
def build_qa_chain(pdf_bytes):
    """Load, chunk, embed and build the QA chain (cached per uploaded file)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    # Load (with OCR fallback), chunk, embed, and store
    try:
        docs = load_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)
    if not docs:
        raise ValueError("No text could be extracted from this PDF, even with OCR.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # chromadb shares one store per process – isolate each uploaded PDF
    # in its own collection keyed by file content
    file_id = hashlib.md5(pdf_bytes).hexdigest()[:12]
    vectordb = Chroma.from_documents(
        chunks,
        embeddings,
        collection_name=f"doc_{file_id}",
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    llm = OllamaLLM(model="llama3.2", temperature=0.2)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )
    return qa_chain, chunks


uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    err = ollama_error()
    if err:
        st.error(err)
        st.stop()

    with st.spinner("Indexing PDF..."):
        try:
            qa_chain, chunks = build_qa_chain(uploaded_file.getvalue())
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Could not process this PDF ({e.__class__.__name__}): {e}")
            st.stop()

    st.success(f"PDF processed ({len(chunks)} text chunks)! Ask a question below.")
    query = st.text_input("Your question:")
    if query.strip():
        with st.spinner("Thinking..."):
            try:
                result = qa_chain.invoke({"query": query.strip()})
            except Exception as e:
                st.error(f"Ollama request failed: {e}")
                st.stop()
        st.write(result["result"])

        with st.expander("Sources"):
            for doc in result.get("source_documents", []):
                page = doc.metadata.get("page")
                label = f"Page {page + 1}" if isinstance(page, int) else "Source"
                st.markdown(f"**{label}**")
                st.text(doc.page_content[:400])
                st.divider()
