# Framing and summarisation outputs

Written by section 6 of `notebooks/05_framing_summarisation/newsprism_summary_no_events.ipynb` on 2026-10-15T07:28:44+00:00 (model: llama3.2:latest). 253 of 253 articles analysed. Do not edit by hand; rerun the export instead.

- `article_id` is the join key for every file and for the other components' outputs.
- These files hold only this component's results. Political leaning, sentiment and entities are not included: join them from the other components' files on `article_id`.
- Display only items whose `status` is `ok`, and quotations whose `quote_status` is `exact`.
- Groups marked `thin` have fewer than 5 articles: show them as having insufficient data.
- Articles on a topic concern different events, so the views describe patterns of coverage, not agreement or disagreement on facts.

## articles.csv (253 rows)

| Column | Meaning |
| --- | --- |
| `article_id` | Join key, as in the shared corpus |
| `topic` | immigration, housing or clean_energy |
| `publisher` | Outlet name |
| `title` | Headline with the outlet suffix removed |
| `date` | Publication date (approximate) |
| `url` | Link to the original article |
| `article_type` | news_report, opinion, explainer, transcript or other (assigned by the model) |
| `main_country` | The country the article is mainly about (assigned by the model) |
| `used_in_topic_views` | True if the article passes the Australia filter and is therefore counted in the topic views |
| `summary` | Neutral summary of 2 to 3 sentences |
| `dominant_frame` | One of the eight frame labels |
| `frame_reason` | One sentence on why the frame fits |
| `quotes_checked` | Number of quotations and key phrases checked against the article |
| `quotes_exact` | Number of those found word for word |
| `model` | Model that produced the summary |

## article_evidence.csv (2467 rows)

| Column | Meaning |
| --- | --- |
| `article_id` | Join key |
| `kind` | key_point (something the article reports) or key_language (a phrase that shows the framing) |
| `text` | The claim, or the phrase |
| `speaker` | For key points, who makes the claim; for key language, narration or quoted_source |
| `quote` | Quotation given by the model as evidence |
| `sentence_id` | Sentence in which the quotation was found, or the one cited if it was not found |
| `sentence` | Full text of that sentence, for display as context |
| `quote_status` | exact, near or missing; display only exact |

## frame_counts.csv (73 rows)

| Column | Meaning |
| --- | --- |
| `topic` | Topic |
| `publisher` | Outlet name |
| `dominant_frame` | Frame label |
| `articles` | Number of the outlet's articles on the topic with this frame |
| `publisher_total` | Number of the outlet's articles on the topic across all frames |
| `thin` | True if publisher_total is below MIN_ARTICLES |

## topic_outlets.json (topics: clean_energy, housing, immigration)

Outlet view. Per topic: overview, outlets (one card each: article counts, thin flag, frame counts, top frame, article types, example articles), outlet_patterns and differences. Show patterns and differences with status 'ok'.

## topic_stances.json (topics: clean_energy, housing, immigration)

Stance view. Per topic: overview, articles_per_stance, outlets_per_stance, stance_summaries (summary, emphasis, key_claims with verified_support quotations and a human_review verdict) and differences. The groups come from the political leaning component's labels. Show claims whose human_review is 'yes'; status 'ok' only means the quotation was found.

## Loading in Streamlit

```python
import json
from pathlib import Path
import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parents[1] / "notebooks" / "05_framing_summarisation" / "Results"  # from app/<file>.py

@st.cache_data
def load_framing():
    read = lambda name: json.loads((DATA / name).read_text(encoding="utf-8")) if (DATA / name).exists() else {}
    return pd.read_csv(DATA / "articles.csv"), pd.read_csv(DATA / "article_evidence.csv"), pd.read_csv(DATA / "frame_counts.csv"), read("topic_outlets.json"), read("topic_stances.json")
```
