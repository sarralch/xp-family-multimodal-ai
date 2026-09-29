"""Assistant 2 — free-form questions over the cosmetic-safety knowledge base."""

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from xp_family.llm import LLMNotConfiguredError
from xp_family.services import DataMissingError, get_services


def assistant_2_view(request: HttpRequest) -> HttpResponse:
    result, error = None, None
    question = request.POST.get("question", "").strip() if request.method == "POST" else ""
    if question:
        try:
            answer = get_services().knowledge.answer(question)
            result = {"response": answer.answer, "sources": answer.contexts}
        except (DataMissingError, LLMNotConfiguredError) as exc:
            error = str(exc)
    return render(request, "assistant2/assistant2.html", {"result": result, "error": error})
