from xp_family.safety import SafetyChecker, Status, parse_ingredient_list, summarize
from xp_family.services import BANNED_FILE, INGREDIENT_INFO_FILE

from .conftest import SAMPLE_DATA


def checker() -> SafetyChecker:
    return SafetyChecker.from_files(SAMPLE_DATA / BANNED_FILE, SAMPLE_DATA / INGREDIENT_INFO_FILE)


def test_parse_strips_label_splits_and_dedupes():
    text = "Ingrédients : Aqua (Water), Glycerin; glycerin\n• Zinc Oxide."
    assert parse_ingredient_list(text) == ["Aqua (Water)", "Glycerin", "Zinc Oxide"]


def test_banned_matches_on_name_synonym_and_cas_case_and_accent_insensitive():
    c = checker()
    assert c.check_one("HYDROQUINONE").status is Status.BANNED
    assert c.check_one("Oxybenzone").status is Status.BANNED  # synonym
    assert c.check_one("106-88-7").status is Status.BANNED  # CAS number
    assert c.check_one("1,2-epoxybutane").status is Status.BANNED  # accent folded
    assert c.check_one("Benzophenone-3").effects == ["Perturbateur endocrinien", "Allergène"]


def test_parenthesised_and_slashed_inci_names_are_resolved():
    c = checker()
    # "(Bergamot)" is dropped, leaving the banned synonym "citrus bergamia oil".
    assert c.check_one("Citrus Bergamia (Bergamot) Oil").status is Status.BANNED
    assert c.check_one("Aloe Vera (Aloe Barbadensis)").status is Status.SAFE
    assert c.check_one("Aqua/Water/Glycerin").status is Status.SAFE


def test_unknown_is_never_reported_as_safe():
    verdict = checker().check_one("Mystery Extract 42")
    assert verdict.status is Status.UNKNOWN


def test_info_flag_false_yields_caution():
    assert checker().check_one("Retinol").status is Status.CAUTION


def test_summary_counts_every_status():
    verdicts = checker().check(["Glycerin", "Retinol", "Hydroquinone", "Foo"])
    assert summarize(verdicts) == {"banned": 1, "caution": 1, "safe": 1, "unknown": 1}
