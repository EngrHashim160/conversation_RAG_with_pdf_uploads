# Conversational RAG with PDF Uploads & Chat History

A Streamlit app for **chatting with your PDFs** using **RAG**. Upload one or more PDFs, the app chunks them, creates **HuggingFace sentence‑transformer** embeddings, indexes them in a **Chroma** vector store, and answers questions with a **Groq** LLM while keeping **chat history** via LangChain's `RunnableWithMessageHistory`.

---

## 🧩 Tech Stack
- **UI:** Streamlit
- **LLM:** `ChatGroq` (e.g., `Gemma2-9b-It`)
- **Embeddings:** `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`)
- **Vector DB:** Chroma (persisted at `./chroma_db`)
- **PDF Loader:** `PyPDFLoader`
- **LangChain Chains:** history‑aware retriever + retrieval chain + stuff documents chain

---

## ✨ Features
- Multiple **PDF upload** (drag & drop)
- **Persistent** Chroma index (`./chroma_db`)
- **Conversation aware** question reformulation
- **Session IDs** to separate histories
- Simple **.env** support for tokens

---


## ⚙️ Setup

1) **Python & Virtualenv**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2) **Environment variables**
Create a `.env` file (or set in your shell):
```bash
HF_TOKEN=your_huggingface_token   # required by HuggingFaceEmbeddings
```
> You will paste your **GROQ_API_KEY** inside the app sidebar when it runs.

> 🔐 **Security tip:** If a real token was ever committed to `.env`, rotate it immediately.

---

## ▶️ Run
From the project root:
```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit (usually http://localhost:8501).

---

## 🖱️ How to Use
1. Enter your **Groq API key** in the input field.
2. (Optional) Set a **Session ID** to keep separate histories.
3. **Upload PDF(s)** via the file uploader.
4. Ask questions in natural language. Answers will use your PDF context + chat history.
5. Your vector index is saved at `./chroma_db` for reuse between runs.

---

## 🧠 How It Works
- **Load PDFs** → `PyPDFLoader`
- **Split** → `RecursiveCharacterTextSplitter`
- **Embed** → `HuggingFaceEmbeddings("all-MiniLM-L6-v2")`
- **Index** → `Chroma(persist_directory="./chroma_db")`
- **Retrieve+Generate** → History‑aware retriever → RAG pipeline → `ChatGroq`
- **Memory** → `RunnableWithMessageHistory` + `ChatMessageHistory`

---

## 🛠 Troubleshooting
- *No answers / poor recall*: ensure PDFs were processed; verify `./chroma_db` exists.
- *Auth errors*: verify GROQ key (in app) and `HF_TOKEN` env var.
- *Module errors*: re‑install with `pip install -r requirements.txt` inside the active venv.
- *Binary deps*: Chroma may need platform‑specific wheels; upgrade `pip` (`pip install -U pip`).

---

## 📜 License
MIT (or your preferred). Replace this with your actual license if different.
