# PDF Q&A with Ollama (Local RAG)

Ask questions about any PDF using local models — no data leaves your machine.

Built with LangChain + Chroma + Ollama (`llama3.2` for answers, `nomic-embed-text` for embeddings).

## Features

- Web UI (Streamlit): upload a PDF and ask questions
- OCR fallback for scanned PDFs (tesseract)
- Per-PDF isolation: uploading a new PDF never contaminates answers from the previous one
- Source pages shown for every answer
- CLI demo script included

## Setup

Requirements: Python 3.10+, [Ollama](https://ollama.com) with `llama3.2` and `nomic-embed-text`, and `tesseract-ocr` (for scanned PDFs).

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Run

```bash
# Web app
.venv/bin/streamlit run app.py

# CLI demo (uses constitution.pdf – place your own PDF and edit the path)
.venv/bin/python rag_demo.py
```

Then upload a PDF in the browser and ask away.

## Notes

- Handwritten PDFs are not supported by classic OCR — answers will be poor.
- The first run downloads nothing extra; everything runs locally.
