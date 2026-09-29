"""Assistant 1 — label scanner: OCR → deterministic XP screening → grounded explanation.

v1 also displayed BERTScore and Hit@k/MRR on every request. Those numbers compared the
answer with the very document the retriever had just returned, so Hit@k and MRR were
always perfect and measured nothing; offline evaluation now lives in eval/.
"""

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from xp_family import prompts
from xp_family.llm import LLMNotConfiguredError
from xp_family.ocr import OCRUnavailableError, extract_text
from xp_family.safety import parse_ingredient_list, summarize
from xp_family.services import DataMissingError, get_services


def dashboard_assistant(request: HttpRequest) -> HttpResponse:
    return render(request, "assistants/dashboard_assistant.html")


def assistant_1_view(request: HttpRequest) -> HttpResponse:
    result, error = None, None
    if request.method == "POST" and request.FILES.get("image"):
        services = get_services()
        try:
            text = extract_text(request.FILES["image"], services.settings)
            ingredients = parse_ingredient_list(text)
            verdicts = services.safety.check(ingredients)
            answer = (
                services.knowledge.answer(
                    prompts.SCAN_QUESTION.format(ingredients=", ".join(ingredients))
                )
                if ingredients
                else None
            )
            result = {
                "extracted_text": text,
                "verdicts": verdicts,
                "summary": summarize(verdicts),
                "response": answer.answer if answer else "Aucun ingrédient lisible.",
                "sources": answer.contexts if answer else [],
            }
        except (OCRUnavailableError, DataMissingError, LLMNotConfiguredError) as exc:
            error = str(exc)
    return render(request, "assistants/assistant1.html", {"result": result, "error": error})
