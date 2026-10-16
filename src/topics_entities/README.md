# topics_entities

Finds **subtopics** inside each broad topic (clean_energy, housing, immigration) and extracts **named entities** (people, organisations, places, groups) from every article. Every output joins to the corpus on `article_id`.

For the reasoning behind every model and setting, how this connects to the other components and the Streamlit app, and the evidence and limitations, see [DOCUMENTATION.md](DOCUMENTATION.md).

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf # entity model (~450 MB)
python -m spacy download en_core_web_sm  # used for keyword lemmas
```

The first run also downloads the embedding model `BAAI/bge-small-en-v1.5` from Hugging Face (~130 MB).

## Run

From Python (input and output are DataFrames; nothing is read from or written to disk):

```python
import pandas as pd
from src.topics_entities import run, write

result = run(pd.read_csv('data/processed/FINAL_newsprism_corpus.csv'))
result.article_subtopics   # see "Outputs" below
write(result, 'data/enriched/topics_entities')   # optional: save the five CSV files
```

From the command line (run from the repository root):

```bash
python -m src.topics_entities --input data/processed/FINAL_newsprism_corpus.csv --output data/enriched/topics_entities
```

About 2.5 minutes on a laptop CPU. Add `--cache-dir <folder>` to reuse embeddings between runs.

**Input columns required:** `article_id`, `topic`, `title`, `publisher`, `clean_text`. A missing column stops the run with a clear error. Other columns are ignored, and the input is never modified. Boilerplate (newsletter and app promos) is removed in memory only.

### New articles (entities only)

`run()` learns subtopics from a whole corpus, so it needs a corpus. For one or a few freshly fetched articles use `extract_entities`, which skips the subtopic step:

```python
import spacy
from src.topics_entities import extract_entities
from src.topics_entities.config import NER_DISABLE, NER_MODEL

nlp = spacy.load(NER_MODEL, disable=NER_DISABLE)   # load once (~5 s), e.g. with st.cache_resource
mentions, articles = extract_entities(new_articles_df, nlp=nlp)   # same columns as run()
```

It needs the same input columns as `run()` (including `topic`). Subtopics for new articles are not supported yet, because that needs an extra "assign" step.

## Outputs

| File | One row per | Columns |
|---|---|---|
| `article_subtopics.csv` | (article, subtopic) | `article_id`, `topic`, `publisher`, `subtopic`, `label`, `n_chunks`, `weight` (share of the article's passages), `is_primary`, `primary_n_chunks`, `outlier_share`, `review_status`, `relevance`, `scope`, `scope_country` |
| `chunk_subtopics.csv` | passage (~100 words) | `chunk_id`, `article_id`, `topic`, `publisher`, `subtopic` (`-1` = fits no subtopic), `label`, `relevance`, `scope`, `scope_country` |
| `subtopics.csv` | subtopic | `topic`, `subtopic`, `status`, `flag`, `label`, `relevance`, `scope`, `scope_country`, `reason`, `suggested_from`, `keywords`, `n_chunks`, `n_articles`, `n_publishers`, `cross_share` |
| `entity_mentions.csv` | entity mention | `article_id`, `topic`, `publisher`, `chunk_id` (`<article_id>_title` for the headline), `in_title`, `surface` (as written), `entity` (normalised name), `type` |
| `article_entities.csv` | (article, entity) | `article_id`, `topic`, `publisher`, `entity`, `type`, `n_mentions` (in the body), `in_title`, `surface_forms` |

**Entity types:** `PERSON`, `ORG` (includes parties, government bodies, companies, media outlets), `PLACE`, `NORP` (nationalities and other groups, e.g. "Australian", "Muslim").

## How to use them correctly

- **Join keys:**
  - `article_id` everywhere
  - `chunk_id` between `entity_mentions` and `chunk_subtopics`
  - **`(topic, subtopic)` for subtopics:** subtopic numbers restart in each topic, so never join on `subtopic` alone
- **Show `label`, not the subtopic number.** Numbers can change if the module is re-run on different data.
- **Australian analysis:** keep `relevance == 'on_topic'` and `scope in ('australian', 'mixed')`. That's 204 of the 253 articles. `scope == 'foreign'` marks overseas coverage, mostly UK and US.
- **Count articles, not mentions.** One long article can mention a name 40 times. Count the articles that mention an entity.
- **To count entities only where an article is on a subtopic,** join `entity_mentions.chunk_id` to `chunk_subtopics` and keep the accepted passages. Counting by whole article also picks up passing mentions in unrelated paragraphs.
- **"Australia" (PLACE) and "Australian" (NORP)** appear in about half of all articles. They're correct, but tell the reader nothing, so consider hiding them in displays.
- **An entity's identity is `(entity, type)`.** "Labour" (the UK party) is deliberately separate from "Australian Labor Party".
- **19 articles have no subtopic** (every passage is `-1`). They still have entities.
- **The labels are human decisions, not model output.** All 16 subtopics were reviewed by hand. One clean-energy subtopic is `off_topic` (Victorian state politics). One is labelled "Mixed" and should not be read as a single theme.

## If the corpus changes

The subtopic labels are tied to the exact set of passages that was reviewed (`review_decisions.csv`). If the data changes, the run still completes, but any subtopic whose passages changed comes out with:
- `status = needs_review` or `unreviewed`
- empty `label`, `relevance` and `scope`
- the closest previous label in `suggested_from`

Ask Ishan to review those subtopics. The review tool is in `review.py`, and it is not part of a normal run.

## Tests

```bash
python -m unittest discover src/topics_entities/tests
```
These need no models or data and run in under a second.

## Dependency note

`spacy-transformers` (needed for the entity model) requires `transformers` 4.x, so this module pins `transformers==4.53.2` and `sentence-transformers==5.7.0`. If another module needs `transformers` 5, talk to Ishan before changing the pins.
