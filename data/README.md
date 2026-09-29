# Data

The real datasets are **not published** in this repository. Put your copies in this
folder (it is git-ignored, except for this README and `sample/`), or point `DATA_DIR`
at another location.

`sample/` holds a tiny **synthetic** dataset with the same schemas. The test suite and
CI run against it, and you can use it to try the app: `DATA_DIR=data/sample`.

| File | Used by | Schema | v1 file name |
|---|---|---|---|
| `knowledge.json` | Assistants 1 & 2 (retrieval) | JSON list of records. Mixed layouts are fine: regulatory rows (`substance_chimique`, `numero_cas`, `conditions_utilisation`, …) and research summaries (`titre`, `recommandations`, …). Every non-empty field is indexed. | `assistants1/73.json` |
| `banned_ingredients.json` | Safety checker + retrieval | Object keyed by ingredient: `{name, cas, synonyms[], effects[], safe_for_xp}` | `banned_ingredients_cleaned.json` |
| `ingredients_info.json` | Safety checker + retrieval | Object keyed by ingredient: `{description, function, good_for, avoid, safe_for_xp}` | `ingredients_info_cleaned.json` |
| `products_xp_checked.json` | Assistant 3 (recipes) | JSON list: `{name, type, ingredients[], safe_for_xp, contains[]}`, where `contains` lists the flagged ingredients | `skincare_recipes_xp_checked.json` |

## How the files are used

- **Safety screening** is deterministic. An ingredient is `banned` if it matches a name,
  synonym or CAS number in `banned_ingredients.json`. Otherwise it is `safe` or `caution`
  according to `safe_for_xp` in `ingredients_info.json`. Anything else is `unknown`.
  Unknown ingredients are shown to the user and are never treated as safe.
- **Retrieval** indexes `knowledge.json`, `banned_ingredients.json` and
  `ingredients_info.json` for questions, and `products_xp_checked.json` for recipes.
- Embeddings are cached under `CACHE_DIR` as `.npy` files, keyed by a fingerprint of the
  embedding model and document IDs, so changing a file rebuilds the cache automatically.
