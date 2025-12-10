from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llama_cpp import Llama
import whisper
import os
import pytesseract
from databases import Database
import shutil
import uuid
from PIL import Image

DATABASE_URL = "postgresql://postgres:parthav7721@localhost:5432/chatbot_db"
database = Database(DATABASE_URL)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

llm = None
whisper_model = None
vector_store = None
conversation_history = []

app = FastAPI(title="Local Chatbot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

@app.on_event("startup")
async def startup():
    global llm, whisper_model, vector_store
    try:
        await database.connect()
        print(" Database connected.")
    except Exception as e:
        print(f" Database connection failed: {e}")

    try:
        llm = Llama(
            model_path="models/openhermes-2.5-mistral-7b.Q4_K_M.gguf",
            n_ctx=2048,
            n_threads=max(1, os.cpu_count() - 1),
            verbose=False,
        )
        print(" LLM loaded.")
    except Exception as e:
        print(f" Error loading LLM: {e}")

    try:
        whisper_model = whisper.load_model("small")
        print(" Whisper loaded.")
    except Exception as e:
        print(f" Error loading Whisper: {e}")

    try:
        print(" Setting up Gale PDF RAG pipeline...")
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        faiss_index_path = "gale_faiss_index"

        if os.path.exists(faiss_index_path):
            vector_store = FAISS.load_local(
                faiss_index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            print(" FAISS vector store loaded.")
        else:
            loader = PyPDFLoader(
                r"C:\\Users\\kavit\\OneDrive\\Clustering\\gen ai tutorial\\The_Gale_Encyclopedia_of_MEDICINE_SECOND.pdf"
            )
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = splitter.split_documents(documents)
            vector_store = FAISS.from_documents(docs, embeddings)
            vector_store.save_local(faiss_index_path)
            print(" FAISS vector store built and saved.")
    except Exception as e:
        print(f" Error setting up FAISS: {e}")

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()
    print(" Database disconnected.")

@app.get("/", response_class=FileResponse)
async def serve_ui():
    if not os.path.exists("UserInterFace.html"):
        raise HTTPException(status_code=404, detail="Frontend file not found.")
    return FileResponse("UserInterFace.html", media_type="text/html")

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    global conversation_history
    if llm is None:
        raise HTTPException(status_code=503, detail="LLM not loaded.")

    question = request.question.strip()
    fallback_used = False

    try:
        query = """
            SELECT input FROM chatbot_logs
            ORDER BY time DESC
            LIMIT 5
        """
        rows = await database.fetch_all(query=query)
        rows = rows[::-1]
        last_user_inputs = "\n".join([row["input"].strip().replace("\n", " ") for row in rows])
    except Exception as e:
        print(f"[CONTEXT RETRIEVAL ERROR] {e}")
        last_user_inputs = ""

    contextualized_query = (last_user_inputs + "\n" + question).strip()

    try:
        relevant_docs = vector_store.similarity_search(contextualized_query, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs]).strip()
    except Exception as e:
        print(f"[RAG ERROR] {e} - falling back to direct prompt.")
        context = ""
        fallback_used = True

    try:
        query = """
            SELECT input, output FROM chatbot_logs
            ORDER BY time DESC
            LIMIT 5
        """
        rows = await database.fetch_all(query=query)
        rows = rows[::-1]
        history_str = ""
        for row in rows:
            user_input = row["input"].strip().replace("\n", " ")
            assistant_output = row["output"].strip().replace("\n", " ")
            history_str += f"User: {user_input}\nAssistant: {assistant_output}\n"
    except Exception as e:
        print(f"[CONTEXT HISTORY ERROR] {e}")
        history_str = ""

    prompt = f"""
You are a helpful, conversational medical assistant. Use the provided medical context to answer clearly, referencing prior conversation naturally if needed.

Context:
{context}

Conversation so far:
{history_str}

User: {question}
Assistant:
""".strip()

    try:
        response = llm(
            prompt,
            max_tokens=512,
            temperature=0.7,
            stop=["User:", "<|end|>"]
        )
        answer = response["choices"][0]["text"].strip()

        conversation_history.append({"question": question, "answer": answer})

        insert_query = """
            INSERT INTO chatbot_logs (input, output, time)
            VALUES (:input, :output, NOW())
        """
        values = {"input": question, "output": answer}
        await database.execute(query=insert_query, values=values)

        return {
            "question": question,
            "answer": answer,
            "fallback_used": fallback_used
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation error: {e}")

@app.post("/speech-to-text")
async def speech_to_text(file: UploadFile = File(...)):
    try:
        with open("temp_audio.wav", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = whisper_model.transcribe("temp_audio.wav")

        os.remove("temp_audio.wav")

        return {"text": result["text"]}
    except Exception as e:
        print(f"[WHISPER ERROR] {e}")
        raise HTTPException(status_code=500, detail="Transcription failed.")

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    try:
        file_ext = os.path.splitext(file.filename)[1].lower()
        allowed_exts = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']

        if file_ext not in allowed_exts:
            raise HTTPException(status_code=400, detail="Unsupported file type for OCR.")

        temp_filename = f"ocr_upload_{uuid.uuid4().hex}{file_ext}"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        image = Image.open(temp_filename)
        extracted_text = pytesseract.image_to_string(image)

        output_file = f"ocr_output_{uuid.uuid4().hex}.txt"
        with open(output_file, "w", encoding="utf-8") as out_f:
            out_f.write(extracted_text)

        os.remove(temp_filename)

        return {"output_file": output_file, "message": "Text extracted and saved."}
    except Exception as e:
        print(f"[OCR ERROR] {e}")
        raise HTTPException(status_code=500, detail="OCR extraction failed.")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
