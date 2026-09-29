"""Assistant 3 — XP-safe homemade recipes.

Fixes from v1: recipes are now retrieved from the XP-annotated product file (v1 read
the raw product export, whose ingredient *string* was joined letter by letter and
whose records had no 'name' field), and ingredients flagged as unsuitable for XP block
generation, not only those on the banned list.
"""

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from xp_family import prompts
from xp_family.llm import LLMNotConfiguredError
from xp_family.safety import Status, parse_ingredient_list
from xp_family.services import DataMissingError, get_services


def assistant_3_view(request: HttpRequest) -> HttpResponse:
    result, error = None, None
    if request.method == "POST":
        ingredients = parse_ingredient_list(request.POST.get("ingredients", ""))
        product_type = request.POST.get("product_type", "").strip()
        services = get_services()
        try:
            verdicts = services.safety.check(ingredients)
            blocked = [v for v in verdicts if v.status in (Status.BANNED, Status.CAUTION)]
            if blocked:
                lines = [
                    f"🚨 {v.ingredient} : {', '.join(v.effects) or 'non adapté à l’XP'}"
                    for v in blocked
                ]
                result = {
                    "response": "Ingrédients non adaptés à l’XP :\n" + "\n".join(lines),
                    "verdicts": verdicts,
                    "sources": [],
                }
            elif ingredients:
                question = prompts.RECIPE_QUESTION.format(
                    product_type=f" de type {product_type}" if product_type else "",
                    ingredients=", ".join(ingredients),
                )
                answer = services.recipes.answer(question)
                result = {
                    "response": answer.answer,
                    "verdicts": verdicts,
                    "sources": answer.contexts,
                }
        except (DataMissingError, LLMNotConfiguredError) as exc:
            error = str(exc)
    return render(request, "assistant3/assistant3.html", {"result": result, "error": error})
