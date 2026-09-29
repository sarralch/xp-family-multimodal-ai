from django.shortcuts import render
from django.http import HttpRequest
import os
import pandas as pd

from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.runnables import RunnableMap
import google.generativeai as genai

# 🔑 Clé API Gemini
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY") or ""

# ✅ Wrapper Gemini
class GeminiLLM(LLM):
    model: str = "gemini-2.0-flash"
    api_key: str = os.getenv("GEMINI_API_KEY")
    _client = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        genai.configure(api_key=self.api_key)
        GeminiLLM._client = genai.GenerativeModel(self.model)

    def _call(self, prompt: str, stop=None) -> str:
        chat = GeminiLLM._client.start_chat()
        response = chat.send_message(prompt)
        return response.text.strip()

    @property
    def _llm_type(self): return "gemini"
    @property
    def _identifying_params(self): return {"model": self.model}

# 📄 Charger les données JSON
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(BASE_DIR, "73.json")
df = pd.read_json(json_path)
data = df.to_dict(orient="records")

# 📄 Documents pour FAISS
docs = [
    Document(page_content="\n".join(f"{k.capitalize()}: {v}" for k, v in item.items()), metadata=item)
    for item in data
]

# 📦 Embedding + FAISS
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_path = os.path.join(BASE_DIR, "ingredients_vector_index")

if os.path.exists(db_path):
    db = FAISS.load_local(folder_path=db_path, embeddings=embedder, allow_dangerous_deserialization=True)
else:
    db = FAISS.from_documents(docs, embedder)
    db.save_local(db_path)

retriever = db.as_retriever(search_kwargs={"k": 3})

# 🧠 Prompt + Chaîne RAG
llm = GeminiLLM()
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Tu es un expert en formulation cosmétique et en toxicologie. Utilise le CONTEXTE pour répondre à la QUESTION en tenant compte des peaux sensibles, notamment celles atteintes de Xeroderma Pigmentosum (XP).

CONTEXTE :
{context}

QUESTION :
{question}

RÉPONSE :
""".strip()
)

rag_chain = RunnableMap({
    "context": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | create_stuff_documents_chain(llm=llm, prompt=prompt, document_variable_name="context")

# 🌐 Vue Django – Assistant 2 (chat textuel uniquement)
def assistant_2_view(request: HttpRequest):
    result = None
    if request.method == "POST":
        question = request.POST.get("question", "")
        if question:
            response = rag_chain.invoke({"question": question})
            result = {"response": response.strip()}

    return render(request, "assistant2/assistant2.html", {"result": result})
