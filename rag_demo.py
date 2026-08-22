import os

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA 

# 1. Load the PDF – change the path!
loader = PDFPlumberLoader("constitution.pdf")   # or your file
documents = loader.load()

# 2. Split into overlapping chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)

# 3. Embed and store in Chroma (persistent)
# Only build once – reuse the existing database on later runs
embeddings = OllamaEmbeddings(model="nomic-embed-text")
if os.path.exists("./chroma_db/chroma.sqlite3"):
    vectordb = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
    )
else:
    vectordb = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./chroma_db"   # saved to disk, will reuse later
    )

# 4. Create a retriever (fetch top 3 relevant chunks) and a QA chain
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
llm = OllamaLLM(model="llama3.2", temperature=0.2)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# 5. Ask a question
question = "What are the first three articles of the Constitution about?"
answer = qa_chain.invoke({"query": question})
print(answer["result"])
