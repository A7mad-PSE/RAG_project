import hashlib
import os
import sys

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from utils import ollama_error

PROMPT = PromptTemplate(
    template=(
        "You are an assistant answering questions about a document.\n"
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

PDF_PATH = "constitution.pdf"

error = ollama_error()
if error:
    sys.exit(error)

# 1. Load the PDF – change the path!
try:
    documents = PDFPlumberLoader(PDF_PATH).load()
except FileNotFoundError:
    sys.exit(f"PDF not found: {PDF_PATH} – update PDF_PATH at the top of this script.")
except Exception as exc:
    sys.exit(f"Failed to read {PDF_PATH}: {exc}")

if not any(d.page_content.strip() for d in documents):
    sys.exit(
        f"No extractable text found in {PDF_PATH}. "
        "Scanned PDFs are only supported in the web app (app.py), which has OCR."
    )

# 2. Split into overlapping chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
chunks = [
    c for c in text_splitter.split_documents(documents) if c.page_content.strip()
]

# 3. Embed and store in Chroma (persistent)
# Reuse the existing database, but rebuild it if the PDF changed
embeddings = OllamaEmbeddings(model="nomic-embed-text")
with open(PDF_PATH, "rb") as f:
    pdf_hash = hashlib.md5(f.read()).hexdigest()
hash_marker = "./chroma_db/.source_hash"
db_exists = os.path.exists("./chroma_db/chroma.sqlite3")
if db_exists and os.path.exists(hash_marker):
    with open(hash_marker) as f:
        stored_hash = f.read().strip()
else:
    stored_hash = None

if db_exists and stored_hash == pdf_hash:
    vectordb = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
    )
else:
    if db_exists:
        print("PDF changed since the last run – rebuilding the vector database...")
        import shutil
        shutil.rmtree("./chroma_db")
    vectordb = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./chroma_db"   # saved to disk, will reuse later
    )
    os.makedirs("./chroma_db", exist_ok=True)
    with open(hash_marker, "w") as f:
        f.write(pdf_hash)

# 4. Create a retriever (fetch top 3 relevant chunks) and a QA chain
retriever = vectordb.as_retriever(search_kwargs={"k": 5})
llm = OllamaLLM(model="llama3.2", temperature=0.2)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT},
)

# 5. Ask a question
question = "How can the Constitution be amended?"
try:
    answer = qa_chain.invoke({"query": question})
except Exception as exc:
    sys.exit(f"Ollama request failed: {exc}")
print(answer["result"])
