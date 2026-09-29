from django.shortcuts import render
from django.http import HttpRequest
import os
import json
import pandas as pd
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.runnables import RunnableMap
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# 🔑 Clé API Gemini
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY") or ""

class GeminiLLM(LLM):
    model: str = "gemini-2.0-flash"
    api_key: str = os.getenv("GEMINI_API_KEY")
    _client = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        genai.configure(api_key=self.api_key)
        GeminiLLM._client = genai.GenerativeModel(self.model)

    def _call(self, prompt: str, stop=None) -> str:
        response = GeminiLLM._client.generate_content(prompt)
        return response.text.strip()

    @property
    def _llm_type(self): return "gemini"
    @property
    def _identifying_params(self): return {"model": self.model}

# 🧪 Chargement des données JSON
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "banned_ingredients_cleaned.json"), "r", encoding="utf-8") as f:
    banned_ingredients = json.load(f)

with open(os.path.join(BASE_DIR, "skincare_products_cleaned.json"), "r", encoding="utf-8") as f:
    recipe_data = json.load(f)

# 🧱 Vectorisation des recettes
documents = []
for recipe in recipe_data:
    name = recipe.get("name", "Recette inconnue")
    ingredients = ", ".join(recipe.get("ingredients", []))
    instructions = recipe.get("instructions", "")
    text = f"Nom: {name}\nIngrédients: {ingredients}\nInstructions: {instructions}"
    documents.append(Document(page_content=text, metadata={"name": name, "ingredients": ingredients}))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
split_docs = text_splitter.split_documents(documents)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(split_docs, embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 🧠 Prompt & RAG
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Tu es un expert en formulation cosmétique.

En te basant sur les RECETTES EXISTANTES ci-dessous, propose une nouvelle recette personnalisée, adaptée aux enfants atteints de Xeroderma Pigmentosum (XP).
Ajoute un nom court et clair à ta recette sur la première ligne, précédé de \"Nom :\"

RECETTES EXISTANTES :
{context}

QUESTION :
{question}

RÉPONSE :
"""
)

llm = GeminiLLM()
stuff_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template, document_variable_name="context")
rag_chain = RunnableMap({
    "context": lambda x: retriever.invoke(x["question"]),
    "question": lambda x: x["question"]
}) | stuff_chain

similarity_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def check_xp_safety(user_ingredients):
    user_ingredients = [i.strip().lower() for i in user_ingredients.split(",") if i.strip()]
    dangerous = []
    for ing in user_ingredients:
        for key, data in banned_ingredients.items():
            if ing == key or ing in data.get("synonyms", []):
                dangerous.append((ing, data.get("effects", [])))
                break
    return user_ingredients, dangerous

# 🧾 Vue Django – Assistant 3 (recette sécurisée pour XP)
def assistant_3_view(request: HttpRequest):
    result = None
    if request.method == "POST":
        ingredients = request.POST.get("ingredients", "")
        product_type = request.POST.get("product_type", "")

        ingredients_cleaned, issues = check_xp_safety(ingredients)

        if issues:
            message = "\n".join(
                [f"\U0001f6a8 {ing} : {', '.join(effects) if effects else 'interdit pour XP'}" for ing, effects in issues]
            )
            result = {"response": f"\n\U0001f6a8 Ingrédients dangereux détectés pour XP :\n{message}"}
        else:
            type_str = f" de type {product_type}" if product_type else ""
            question = f"Propose une recette maison{type_str}, à base de : {', '.join(ingredients_cleaned)}, pour un enfant atteint de XP."
            try:
                generated = rag_chain.invoke({"question": question})
                retrieved = retriever.invoke(question)
                top_recipe = retrieved[0].page_content
                original_instr = retrieved[0].metadata.get("ingredients", "")
                embeddings = similarity_model.encode([generated.strip(), original_instr])
                score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                similarity_pct = round(score * 100, 2)

                response = f"🤖 Recette IA générée :\n{generated.strip()}\n\n📘"
                result = {"response": response}
            except Exception as e:
                result = {"response": f"Erreur RAG Gemini : {e}"}

    return render(request, "assistant3/assistant3.html", {"result": result})
