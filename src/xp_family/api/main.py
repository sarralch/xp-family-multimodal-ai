"""FastAPI backend exposing the three assistants as a JSON API.

uvicorn xp_family.api.main:app --reload     # docs at http://localhost:8000/docs
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from xp_family import __version__, prompts
from xp_family.config import get_settings
from xp_family.llm import LLMNotConfiguredError
from xp_family.observability import configure_logging, new_trace_id
from xp_family.ocr import OCRUnavailableError, extract_text
from xp_family.rag.pipeline import RAGAnswer
from xp_family.safety import Status, Verdict, parse_ingredient_list, summarize
from xp_family.services import DataMissingError, Services, get_services

DISCLAIMER = "Informations éducatives : elles ne remplacent pas l'avis d'un dermatologue."

settings = get_settings()
configure_logging(settings.log_level, settings.log_json)

app = FastAPI(
    title="XP Family Support API",
    version=__version__,
    description="Cosmetic safety screening, ingredient Q&A and XP-safe recipes for "
    "Xeroderma Pigmentosum patients. " + DISCLAIMER,
)

ServicesDep = Annotated[Services, Depends(get_services)]


@app.middleware("http")
async def add_trace_id(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    trace_id = new_trace_id()
    response = await call_next(request)
    response.headers["X-Trace-Id"] = trace_id
    return response


@app.exception_handler(DataMissingError)
@app.exception_handler(LLMNotConfiguredError)
async def _not_configured(_: Request, exc: Exception) -> Response:
    """Missing data files or API key: a deployment problem, reported as 503 not 500."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# --- Schemas -----------------------------------------------------------------


class Source(BaseModel):
    id: str
    title: str
    score: float
    excerpt: str


class VerdictOut(BaseModel):
    ingredient: str
    status: Status
    matched: str | None = None
    effects: list[str] = []


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class AskOut(BaseModel):
    answer: str
    sources: list[Source]
    disclaimer: str = DISCLAIMER


class SafetyIn(BaseModel):
    ingredients: str = Field(min_length=1, max_length=10_000, examples=["Aqua, Glycerin"])


class SafetyOut(BaseModel):
    verdicts: list[VerdictOut]
    summary: dict[str, int]
    disclaimer: str = DISCLAIMER


class ScanOut(SafetyOut):
    extracted_text: str
    answer: str | None
    sources: list[Source]


class RecipeIn(BaseModel):
    ingredients: str = Field(min_length=1, max_length=2000)
    product_type: str | None = Field(None, max_length=100)


class RecipeOut(BaseModel):
    verdicts: list[VerdictOut]
    recipe: str | None
    refused_reason: str | None = None
    sources: list[Source]
    disclaimer: str = DISCLAIMER


def _sources(result: RAGAnswer) -> list[Source]:
    return [
        Source(
            id=h.document.id,
            title=h.document.title,
            score=round(h.score, 4),
            excerpt=h.document.text[:300],
        )
        for h in result.contexts
    ]


def _verdicts(verdicts: list[Verdict]) -> list[VerdictOut]:
    return [VerdictOut(**v.__dict__) for v in verdicts]


# --- Routes ------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/v1/ask", response_model=AskOut)
def ask(body: AskIn, services: ServicesDep) -> AskOut:
    """Assistant 2 — free-form question over the cosmetic-safety knowledge base."""
    result = services.knowledge.answer(body.question)
    return AskOut(answer=result.answer, sources=_sources(result))


@app.post("/v1/safety-check", response_model=SafetyOut)
def safety_check(body: SafetyIn, services: ServicesDep) -> SafetyOut:
    """Deterministic screening of an ingredient list (no LLM involved)."""
    verdicts = services.safety.check(parse_ingredient_list(body.ingredients))
    return SafetyOut(verdicts=_verdicts(verdicts), summary=summarize(verdicts))


@app.post("/v1/scan", response_model=ScanOut)
def scan(
    services: ServicesDep, image: Annotated[UploadFile, File(description="Label photo")]
) -> ScanOut:
    """Assistant 1 — OCR a product label, screen each ingredient, then explain the risks."""
    try:
        text = extract_text(image.file, services.settings)
    except OCRUnavailableError as exc:
        raise HTTPException(503, detail=str(exc)) from exc
    ingredients = parse_ingredient_list(text)
    verdicts = services.safety.check(ingredients)
    if not ingredients:
        return ScanOut(
            extracted_text=text, verdicts=[], summary=summarize([]), answer=None, sources=[]
        )
    result = services.knowledge.answer(
        prompts.SCAN_QUESTION.format(ingredients=", ".join(ingredients))
    )
    return ScanOut(
        extracted_text=text,
        verdicts=_verdicts(verdicts),
        summary=summarize(verdicts),
        answer=result.answer,
        sources=_sources(result),
    )


@app.post("/v1/recipe", response_model=RecipeOut)
def recipe(body: RecipeIn, services: ServicesDep) -> RecipeOut:
    """Assistant 3 — XP-safe homemade recipe; refuses if any ingredient is banned/unsuitable."""
    ingredients = parse_ingredient_list(body.ingredients)
    verdicts = services.safety.check(ingredients)
    blocked = [v for v in verdicts if v.status in (Status.BANNED, Status.CAUTION)]
    if blocked:
        names = ", ".join(v.ingredient for v in blocked)
        return RecipeOut(
            verdicts=_verdicts(verdicts),
            recipe=None,
            refused_reason=f"Ingrédients non adaptés à l'XP : {names}",
            sources=[],
        )
    product_type = f" de type {body.product_type}" if body.product_type else ""
    question = prompts.RECIPE_QUESTION.format(
        product_type=product_type, ingredients=", ".join(ingredients)
    )
    result = services.recipes.answer(question)
    return RecipeOut(verdicts=_verdicts(verdicts), recipe=result.answer, sources=_sources(result))


def run() -> None:  # console script: xp-family-api
    import uvicorn

    uvicorn.run("xp_family.api.main:app", host="0.0.0.0", port=8000)
