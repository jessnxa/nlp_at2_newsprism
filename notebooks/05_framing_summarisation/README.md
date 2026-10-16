# 05 Framing and summarisation

This component summarises each article in the NewsPrism corpus, labels its dominant frame, and compares coverage of immigration, housing and clean energy across outlets and across political stance groups. Everything is generated offline with local LLMs; the application only reads the exported files in `Results/`.

## What it produces

| Output | Description |
| --- | --- |
| Article summaries | For each of the 253 articles: a short summary, key points with supporting quotations, key phrases and a dominant frame |
| Outlet view | For each topic: what each outlet's coverage tends to emphasise, and how outlets differ |
| Stance view | For each topic: what coverage labelled left, centre or right reports, with each key claim tied to a quotation |

Comparison is at the topic level only. Almost every event in the corpus has a single article, so outlets' reports of the same event cannot be compared.

## Folder contents

| Path | Contents |
| --- | --- |
| `newsprism_summary_no_events.ipynb` | The whole pipeline: data checks, baselines, LLM analysis, evaluation and export |
| `Results/` | Final results for the application (see below) |
| `outputs/` | Working files: saved LLM results (`article_analysis.json`, `topic_analysis.json`, `stance_analysis.json`) and the human review sheets |
| `stance_labels.csv` | Political leaning labels used to form the stance groups |
| `figures/` | Saved figures |

## Results for the application

`Results/` holds only this component's own output. Political leaning, sentiment and entities are not copied into it; join them from the other components' results on `article_id`.

| File | One row or entry per | Contents |
| --- | --- | --- |
| `articles.csv` | article | Summary, article type, main country and dominant frame |
| `article_evidence.csv` | key point or key phrase | The claim, its quotation, the source sentence and whether the quotation was verified |
| `frame_counts.csv` | topic, outlet and frame | Frame counts by outlet, in long format for charting |
| `topic_outlets.json` | topic | Outlet view: overview, one card and pattern per outlet, differences between outlets |
| `topic_stances.json` | topic | Stance view: overview, one summary per stance group with key claims and quotations, differences between groups |
| `manifest.json`, `README.md` | export | Settings, row counts and a description of every column, with a loading example |

When displaying these:

- Show only items whose `status` is `ok`, and quotations whose `quote_status` is `exact`.
- Treat groups marked `thin` (fewer than 5 articles) as insufficient data.
- Each stance key claim has a `human_review` verdict. Present only claims marked `yes` as supported.

## Using the results in the Streamlit app

The app needs no model, API key or notebook code. It only reads the files in `notebooks/05_framing_summarisation/Results/`, and `article_id` is the join key throughout.

**Loading**
- Read the three CSV files with pandas and the two JSON files with `json`. Wrap the loading in `@st.cache_data` so it happens once per session.
- The JSON files are keyed by topic: `immigration`, `housing`, `clean_energy`.
- Column meanings are listed in `Results/README.md` and `Results/manifest.json`.

**Suggested views**

| View | Files | What to show |
| --- | --- | --- |
| Article | `articles.csv`, `article_evidence.csv` | For a chosen article: `summary`, `dominant_frame`, a link to `url`, and its key points with quotations (`kind == "key_point"`). Framing phrases are `kind == "key_language"` |
| Outlets | `topic_outlets.json`, `frame_counts.csv` | For a chosen topic: `overview`, a chart of frame counts by outlet, and each entry of `outlet_patterns` and `differences`. Each outlet's card in `outlets` also has example articles |
| Stances | `topic_stances.json` | For a chosen topic: `overview`, then for each entry of `stance_summaries` its `summary` and `key_claims`. Group sizes are in `articles_per_stance` |

**Display rules**
- Show a quotation only if its `quote_status` is `exact`.
- Show a pattern, difference or stance summary only if its `status` is `ok`.
- Leave outlets and groups marked `thin` off charts and say they have too few articles.
- Show a stance key claim only if its `human_review` is `yes`. Currently 2 of 15 qualify, so most groups will show none; say so instead of falling back to unreviewed claims.
- Label the outlet and stance text as model-generated. Only the stance key claims have been checked by a person.

**Other components.** Political leaning, sentiment and entities are not in these files. Load them from the other components' own results and merge on `article_id`.

**Refreshing.** After the notebook's export is re-run, restart the app or clear its cache to pick up the new files.

## How it works

1. **Preparation.** Each article is cleaned and split into numbered sentences, so the model can cite a sentence ID with every quotation.
2. **Baselines.** Lead-3 and a TF-IDF centroid summary give a point of comparison.
3. **Article analysis.** One LLM call per article returns JSON that is validated against a schema. Every quotation is then searched for in the article and marked exact, near or missing.
4. **Outlet and stance views.** One LLM call per topic works from the article summaries and verified quotations, not the full text. Cited article IDs and quotations are checked again.
5. **Evaluation.** Automatic checks, ROUGE against ABC's "In short" summaries, and a human review.
6. **Export.** The saved results are written to `Results/`.

## Running the notebook

Inputs, all joined by `article_id`:

| Input | Path |
| --- | --- |
| Corpus | `data/processed/FINAL_newsprism_corpus.csv` |
| Political leaning labels | `notebooks/05_framing_summarisation/stance_labels.csv` |
| Sentiment results | `notebooks/03_sentiment_language/Results/sentiment_analysis_results.csv` |
| Named entities | `data/enriched/topics_entities/article_entities.csv` |

Setup:

1. Install the packages in the repository's `requirements.txt`.
2. Install [Ollama](https://ollama.com), start it, and download the two models:
   ```bash
   ollama pull llama3.2
   ollama pull llama3.1:8b
   ```
3. Open the notebook and run all cells.

The saved results in `outputs/` are reused, so a normal run makes no model calls and finishes in a few minutes. A cell only calls a model when its saved result is missing or out of date. Regenerating everything from scratch takes several hours on a 16 GB laptop: `llama3.2` handles the article summaries and the outlet view, and `llama3.1:8b` handles the stance view. On macOS the notebook keeps the computer awake while it runs.

Changing `MODEL` or `PROMPT_VERSION` in the notebook invalidates the saved results and triggers a full regeneration.

## Evaluation summary

A human review of 30 randomly sampled summaries (10 per topic) and all 15 stance key claims found:

| Measure | Result |
| --- | --- |
| Faithfulness of summaries | mean 4.70 of 5 |
| Coherence | mean 4.57 of 5 |
| Coverage | mean 3.13 of 5 |
| Frame label judged reasonable | 22 of 30 |
| Stance claims passing the automatic quotation check | 14 of 15 |
| Stance claims supported on reading | 2 of 15 |

The summaries are faithful but thin, and the automatic quotation check overstates how well the stance claims are supported. Malformed records found in the review were regenerated; the scores above describe the output before that repair. Section 5 of the notebook has the detail.

## Main limitations

- The comparison is by topic, not by event, so differences between outlets mix story selection with framing.
- Frame labels come from a small local model without frame definitions; `security` is over-assigned.
- The stance groups use LLM-generated political leaning labels, and most stance key claims were not supported by their quotations on human review.
- The human review is small: one reviewer, 30 articles and 15 claims.

Section 7 of the notebook lists these in full.
