from django.shortcuts import render
from django.http import HttpRequest
import os
import json
import pandas as pd
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score as bert_score
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
import google.generativeai as genai

def dashboard_assistant(request):
    return render(request, 'assistants/dashboard_assistant.html')

# 📌 Configurer Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# 📌 Clé API Gemini
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY") or ""

# 🚀 Wrapper Gemini LLM
class GeminiLLM(LLM):
    model: str = "gemini-1.5-flash-latest"
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
    def _llm_type(self):
        return "gemini"

    @property
    def _identifying_params(self):
        return {"model": self.model}

# 📂 Charger les données JSON
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(BASE_DIR, "73.json")
if not os.path.exists(json_path):
    raise FileNotFoundError("Le fichier 73.json est introuvable")

df = pd.read_json(json_path)
data = df.to_dict(orient="records")

# 📑 Conversion en Documents
docs = []
for item in data:
    text_parts = []
    for k, v in item.items():
        text_parts.append(f"{k.replace('_', ' ').capitalize()}: {v}")
    doc_text = "\n".join(text_parts)
    docs.append(Document(page_content=doc_text, metadata=item))

# 🧠 Embedding + Base vectorielle
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_path = os.path.join(BASE_DIR, "ingredients_vector_index")

if os.path.exists(db_path):
    db = FAISS.load_local(folder_path=db_path, embeddings=embedder, allow_dangerous_deserialization=True)
else:
    db = FAISS.from_documents(docs, embedder)
    db.save_local(db_path)

retriever = db.as_retriever(search_kwargs={"k": 3})

# 📝 Préparation du RAG
llm = GeminiLLM()
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""Tu es un expert en formulation cosmétique et en toxicologie. Utilise le CONTEXTE pour répondre à la QUESTION en tenant compte des peaux sensibles, notamment celles atteintes de Xeroderma Pigmentosum (XP).

CONTEXTE :
{context}

QUESTION :
{question}

RÉPONSE :"""
)

stuff_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template, document_variable_name="context")
rag_chain = RunnableMap({
    "context": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | stuff_chain

# 🔥 Fonction d'évaluation avancée (Génération + Retrieval)
def evaluate_rag_advanced(reference_text, generated_text, question, retriever, top_k=3):
    results = {}

    # 1. Semantic Similarity
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode([reference_text, generated_text])
    semantic_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    results['Semantic Similarity (%)'] = round(semantic_sim * 100, 2)

    # 2. BERTScore
    P, R, F1 = bert_score([generated_text], [reference_text], lang="en", verbose=False)
    results['BERTScore Precision'] = round(P.mean().item(), 4)
    results['BERTScore Recall'] = round(R.mean().item(), 4)
    results['BERTScore F1'] = round(F1.mean().item(), 4)

    # 3. Retrieval Metrics
    retrieved_docs = retriever.invoke(question)
    retrieved_contents = [doc.page_content for doc in retrieved_docs]
    relevant_doc = reference_text

    hit_at_k = any(doc_text.strip() == relevant_doc.strip() for doc_text in retrieved_contents)
    recall_at_k = 1 if hit_at_k else 0
    precision_at_k = (1 if hit_at_k else 0) / len(retrieved_contents) if retrieved_contents else 0

    try:
        rank = next(i+1 for i, doc_text in enumerate(retrieved_contents) if doc_text.strip() == relevant_doc.strip())
        mrr = 1 / rank
    except StopIteration:
        mrr = 0

    results['Precision@k'] = round(precision_at_k, 4)
    results['Recall@k'] = recall_at_k
    results['Hit@k'] = 1 if hit_at_k else 0
    results['MRR'] = round(mrr, 4)

    return results

# 🎯 Vue Assistant 1
def assistant_1_view(request: HttpRequest):
    result = None
    if request.method == "POST" and request.FILES.get("image"):
        image = Image.open(request.FILES['image'])
        extracted_text = pytesseract.image_to_string(image)
        question = f"Voici les ingrédients extraits : {extracted_text}. Indique les risques pour un enfant atteint de Xeroderma Pigmentosum."

        response = rag_chain.invoke({"question": question}).strip()

        retrieved_doc = retriever.invoke(question)[0]
        effet_reference = retrieved_doc.page_content

        # ⚡ Evaluation avancée
        evaluation_results = evaluate_rag_advanced(effet_reference, response, question, retriever)

        result = {
            "extracted_text": extracted_text.strip(),
            "response": response,
            "bert_precision": evaluation_results['BERTScore Precision'],
            "bert_recall": evaluation_results['BERTScore Recall'],
            "bert_f1": evaluation_results['BERTScore F1'],
            "recall_at_k": evaluation_results['Recall@k'],
            "hit_at_k": evaluation_results['Hit@k'],
            "mrr": evaluation_results['MRR'],
        }

    return render(request, "assistants/assistant1.html", {"result": result})














