# Topics and entities: component documentation

Reference for the topic modelling and named entity component of NewsPrism (`src/topics_entities`): what it chose, why, how it connects to the other components and the Streamlit app, and how to use and maintain it. Short usage notes are in `README.md`; this file holds the reasoning and evidence.

## 1. At a glance

This component answers two questions about every article: **what is it about** (which of 16 human-labelled subtopics) and **who and what does it mention** (people, organisations, places, groups). It takes the team's processed corpus and returns five tables that join to everything else on `article_id`.

| | |
|---|---|
| **Owner** | Ishaanjot Singh (branch `ishaan_subtopic_and_entity`, folder `src/topics_entities`) |
| **Input** | A DataFrame with `article_id`, `topic`, `title`, `publisher`, `clean_text` |
| **Output** | `article_subtopics`, `chunk_subtopics`, `subtopics`, `entity_mentions`, `article_entities` |
| **Models** | Embeddings: `BAAI/bge-small-en-v1.5`. Clustering: UMAP + HDBSCAN. Keywords: BERTopic c-TF-IDF. Entities: spaCy `en_core_web_trf`. |
| **Scale (final corpus)** | 253 articles → 2,107 passages → 16 subtopics (all human-reviewed) and 11,455 entity mentions of 3,030 distinct entities |
| **Run time** | About 2.5 minutes on a laptop CPU |
| **Deterministic** | Yes: fixed seed 42, and a clean install reproduced every output |
| **Entry points** | `run(corpus)` for a whole corpus, `extract_entities(articles)` for newly fetched articles |

**What it does not do.** It assigns no political lean, sentiment or framing. Those belong to other components, and keeping them out stops lean labels from leaking into topics and entities.

## 2. How it fits in NewsPrism

The component reads one shared corpus and writes five tables. It modifies nothing else, so every other component keeps working on the original `clean_text`.

```text
Processed corpus (Jessica) --> topics_entities (this component) --> five tables --+
data/processed/                 Subtopics: clean text, ~100-word                   |
FINAL_newsprism_corpus.csv      passages, bge-small embeddings,                    |
                                UMAP + HDBSCAN, 16 labelled by hand                +--> Streamlit app (Lynette)
                                Entities: spaCy trf, name rules,                   |    joins everything on
                                type voting, passage IDs                           |    article_id
Political leaning (Hayden) --------------------------------------------------------+
Sentiment, language (Thimaya) -----------------------------------------------------+
Framing, summaries (Robin) --------------------------------------------------------+
```

The cleaned text exists only in memory during a run. Everything is keyed on `article_id`, and passage-level tables also carry `chunk_id`.

| Connected to | Direction | Join key | What flows |
|---|---|---|---|
| **Data preprocessing (Jessica)** | In | `article_id` | `article_id`, `topic`, `title`, `publisher`, `clean_text` from `data/processed/FINAL_newsprism_corpus.csv` |
| **Political leaning (Hayden)** | Joined to ours | `article_id` | Her `political_label` (left, centre or right) per article, joined onto our subtopics and entities by the app. For a new article, her `predict_political_leaning(text)` returns a label and a confidence. |
| **Sentiment and language (Thimaya)** | Joined to ours | `article_id` | Her sentiment and language outputs, joined the same way (their format is documented by that component) |
| **Framing and summarisation (Robin)** | Can use ours | `article_id`, `chunk_id` | The project plan has framing use article data and relevant NLP outputs, so subtopic labels and entities are available as context |
| **Streamlit app (Lynette)** | Out | `article_id`, `chunk_id`, `(topic, subtopic)` | Subtopic filters and labels, entity counts, and entities for newly fetched articles |

## 3. Models and hyperparameters

All values live in one file, `config.py`, plus the clustering functions in `topics.py`. Nothing is meant to be tuned by callers. Each choice below was made on evidence from this corpus; the experiments are summarised in section 8.

### 3.1 Subtopic pipeline

| Stage | Model or method | Setting | Why |
|---|---|---|---|
| **Cleaning** | Rule-based boilerplate removal (in memory only) | Drop sentences repeated in 3 or more articles (over 3 words), short sentences (30 words or fewer) containing newsletter, sign up, subscribe or app-download wording, and known inline promos | Left in, the topic model made "newsletter / sign up" subtopics and NER counted outlet names as organisations. 2,333 of 241,546 words (1%), but concentrated in SMH, The Age, SBS and 7NEWS. |
| **Chunking** | Sentence-aware fixed windows | Target 100 words, hard cap 190, minimum tail 50, no overlap | The median article is about 1,038 tokens against a 256–512 token embedding limit, so whole-article embeddings would ignore about 75% of the text. The natural idea length on this corpus has a median of 86 words. Overlap would double-count words in c-TF-IDF. |
| **Embedding** | `BAAI/bge-small-en-v1.5` (384 dimensions), normalised | Batch size 32 | Best configuration in a 45-run grid (100-word chunks + bge-small): NPMI 0.134, stability 0.765, coverage 0.923, 0.4 artefact clusters per run. Small and fast on CPU. |
| **Dimension reduction** | UMAP | 5 components, `min_dist` 0.0, cosine metric, `n_neighbors` = `min_cluster_size`, `random_state` 42 | Standard settings from the BERTopic reference setup, not tuned here (the grid varied chunking, embedding model and seed only). A `min_dist` of 0 packs points tightly so density clustering can find cores. |
| **Clustering** | HDBSCAN, run separately for each broad topic | `min_cluster_size` = max(5, round(N / 30)) where N = passages in the topic (19–27 here); `min_samples` 5; Euclidean metric on the UMAP output; `cluster_selection_method` `eom`; outliers kept as `-1` | Expecting 4–10 subtopics, the smallest real one should be about a third of the average size at K = 10. HDBSCAN does not force every passage into a cluster, and generic text ("the minister said") stays out of the subtopic mixes. |
| **Keywords** | BERTopic class-based TF-IDF over our own cluster labels (BERTopic does no clustering here) | 1–3-word phrases, `min_df` 3, English + news stopwords; 30 candidates re-ranked by maximal marginal relevance (diversity 0.1) to 10 | Keywords are evidence for the human label, never the label. MMR at 0.3 dropped defining words such as "rent" and "prices", so 0.1 was used. |
| **Seed** | `SEED = 42` | UMAP, KMeans and sampling | Reproducible runs |

### 3.2 Safeguards, each added because of a failure found in this corpus

| Safeguard | Rule | Failure it prevents |
|---|---|---|
| **Per-article cap (S2)** | Fit clustering on at most `min_cluster_size // 2` evenly spaced passages per article; assign the rest afterwards with HDBSCAN's `approximate_predict` | Long features (up to 42 passages, median 7) formed single-article "subtopics", such as the PNG refugee feature |
| **Split oversized clusters (S3)** | A cluster holding more than 35% of its topic is split with KMeans (k = 2, 3 or 4, best silhouette score), kept only if every part has at least 5 articles and no article supplies more than 50% of a part | A catch-all housing cluster held 49% of passages and merged three real themes |
| **Cross-topic annotation (S1)** | Each passage is marked with the broad topic whose centroid is nearest; clusters above 50% cross-topic are flagged for review. **Annotation only, never a filter.** | Similarity says which topic a passage is nearest, not whether it is relevant: "migration raises rental demand" is genuine immigration coverage |
| **Artefact flags** | A subtopic with fewer than 5 articles, or one article above 50%, is flagged | Single-article clusters |

### 3.3 Named entities

| Item | Choice | Why |
|---|---|---|
| **Model** | spaCy `en_core_web_trf` (transformer), with the tagger, parser, attribute ruler and lemmatiser disabled | F1 0.931 on the hand-annotated gold set against 0.769 for `lg` and 0.712 for `sm`. The selection rule was fixed before scoring. |
| **Kept labels** | `PERSON`, `ORG`, `GPE` and `LOC` merged into `PLACE`, `NORP` | The comparison needs actors and geography. Dates, money and numbers are dropped. |
| **Unit of text** | The same ~100-word passages as the subtopic model, plus each headline | Every mention carries a `chunk_id`, so entities can be counted only in passages of accepted subtopics |
| **Keyword lemmas** | `en_core_web_sm` | A small model is enough to lemmatise keywords |
| **Dependency pins** | `transformers==4.53.2`, `sentence-transformers==5.7.0` | `spacy-transformers` needs `transformers` 4.x. Re-encoding every passage on the pinned versions gave identical topic outputs. |

## 4. Entity rules and human review

### 4.1 Rules applied after the entity model

The model's raw output is noisy in ways specific to news. These fixed, exact-match rules run after it. Each one was verified in context on the corpus, and every rule is shared by the pipeline and the accuracy evaluation.

| Rule | Example | Why |
|---|---|---|
| **Normalise names** | "Ms Hanson's" → "Hanson"; "the Reserve Bank" → "Reserve Bank" | One entity was split across many spellings (Hanson: 80 + 32 + 11 + 5 mentions) |
| **Resolve surnames** | "Hanson" → "Pauline Hanson" only when exactly one full name in the same article ends in it | Merges within an article without guessing across articles; "Hanson-Young" never becomes "Hanson" |
| **Aliases** | RBA → Reserve Bank of Australia; NSW → New South Wales; US → United States | Same entity, different spellings. **"Labour" (UK party) stays separate from "Labor".** |
| **Vote the type** | Majority within each article, then one vote per article, then verified overrides | spaCy tagged "Trump" as ORG 64 times and "NSW" as ORG 60 times. One vote per article stops a long article deciding a type. |
| **Drop self-mentions** | "told SBS News" in an SBS article | The outlet's own name describes the publisher, not the story (152 mentions) |
| **Drop generic institutions** | "the Government", "Cabinet", "the Commonwealth" | Unnamed institutions are not specific entities |
| **Job-title context** | "Energy" in "Energy Minister Chris Bowen" is dropped; "Albanese government" becomes one ORG | Portfolio words were being counted as organisations |
| **Merge plural groups** | "Australians" → "Australian" | Both name the same group (exceptions such as Swiss and Greens are kept) |
| **Short organisation forms** | "Oakes" → "Oakes Energy Services" when exactly one longer name exists | Same entity, shortened |

Two things are deliberately **not** done: no entity is given a political lean, and "Australia" and "Australian" are not removed from the data. They are correct, and they appear in about half of all articles, so apps should hide them by default (section 6).

### 4.2 Human review of the 16 subtopics

Every subtopic was reviewed by a person from its keywords, article list and three kinds of example passages (central, random, cross-topic). A review **names and tags** a subtopic; it never moves a passage or article between subtopics. The tags are copied to every article in that subtopic.

| Field | Values | Rule used |
|---|---|---|
| `label` | A short human name | Says what the articles are about, not just the keywords |
| `relevance` | `on_topic` / `off_topic` | Off-topic when most passages are not about the broad topic |
| `scope` | `australian` / `foreign` / `mixed` | Where the events and policy are, by majority of articles |
| `scope_country` | A country | Only when the scope is not Australian |
| `reason` | One sentence | Records known impurities instead of patching them by hand |

**Outcome:** 15 subtopics are on-topic and 1 is off-topic (`clean_energy` 2, the Victorian politics and VNI West line, about 13 of 40 passages were about energy). Scope: 11 Australian (including the off-topic one), 2 mixed, 3 foreign. **204 of 253 articles** are usable for an Australian comparison (on-topic, Australian or mixed scope), 36 are in foreign on-topic subtopics, and 19 fit no subtopic.

**Decisions are saved against the exact set of passages that was reviewed** (`review_decisions.csv`). If the data changes and a subtopic's passages differ, it comes out as `needs_review` with the closest earlier label suggested (30% article overlap or more), and nothing blocks.

## 5. Output contract

Five tables, written as UTF-8 CSV (with a byte-order mark, so Excel opens them cleanly) by `write()` or the command line. `run()` returns the same five as DataFrames.

| Table | One row per | Columns |
|---|---|---|
| `article_subtopics` | (article, subtopic) | `article_id`, `topic`, `publisher`, `subtopic`, `label`, `n_chunks`, `weight`, `is_primary`, `primary_n_chunks`, `outlier_share`, `review_status`, `relevance`, `scope`, `scope_country` |
| `chunk_subtopics` | passage (~100 words) | `chunk_id`, `article_id`, `topic`, `publisher`, `subtopic` (`-1` = fits no subtopic), `label`, `relevance`, `scope`, `scope_country` |
| `subtopics` | subtopic | `topic`, `subtopic`, `status`, `flag`, `label`, `relevance`, `scope`, `scope_country`, `reason`, `suggested_from`, `keywords`, `n_chunks`, `n_articles`, `n_publishers`, `cross_share` |
| `entity_mentions` | entity mention | `article_id`, `topic`, `publisher`, `chunk_id` (`<article_id>_title` for the headline), `in_title`, `surface` (as written), `entity` (normalised name), `type` |
| `article_entities` | (article, entity) | `article_id`, `topic`, `publisher`, `entity`, `type`, `n_mentions` (in the body), `in_title`, `surface_forms` |

**Key columns**

| Column | Meaning |
|---|---|
| `weight` | Share of the article's clustered passages that fall in this subtopic. An article's weights sum to 1. |
| `is_primary` | This is the article's main subtopic (most passages; ties go to the one that appears first, because news puts the main angle up front) |
| `outlier_share` | Share of the article's passages that fit no subtopic |
| `relevance`, `scope`, `scope_country`, `label` | The human review tags, copied from the subtopic |
| `type` | `PERSON`, `ORG` (parties, government bodies, companies, outlets), `PLACE`, `NORP` (nationalities and other groups) |

**Join rules**

- `article_id` joins every table to the corpus and to the other components' outputs.
- `chunk_id` joins `entity_mentions` to `chunk_subtopics`.
- **Subtopic numbers restart in each broad topic.** Join on `(topic, subtopic)`, never on `subtopic` alone.
- **Show `label`, not the number.** Numbers can change if the module is re-run on different data.
- An entity's identity is `(entity, type)`.
- **Count articles, not mentions:** one long article can mention a name 40 times.
- To count entities only where an article is on a subtopic, filter `entity_mentions` to chunks in accepted subtopics. On the final corpus, 31% of body mentions fall outside the accepted passages, so article-level and passage-level counts differ.

**Coverage.** 234 of 253 articles have a subtopic, and 253 of 253 have at least one entity. A `subtopics` row can be `status = needs_review` or `unreviewed` if the data has changed since the review.

## 6. Using it in the Streamlit app

The app (integration component) reads the five tables and joins them to the other components' outputs on `article_id`. The recipes below were run against the real outputs and Hayden's real lean predictions.

### 6.1 Load once

```python
import pandas as pd
import streamlit as st

@st.cache_data
def load_tables(folder='data/enriched/topics_entities'):
    read = lambda name: pd.read_csv(f'{folder}/{name}.csv', encoding='utf-8-sig')
    return {n: read(n) for n in ['article_subtopics', 'chunk_subtopics', 'subtopics',
                                 'entity_mentions', 'article_entities']}
```

### 6.2 Which articles to show

```python
t = load_tables()
main = t['article_subtopics'][t['article_subtopics'].is_primary]      # one row per article
aus = main[(main.relevance == 'on_topic') & main.scope.isin(['australian', 'mixed'])]
```

This gives 202 articles. (204 articles have *any* on-topic Australian or mixed subtopic; 202 have it as their main subtopic. Pick one definition and keep it everywhere.)

### 6.3 Top entities for a selection

```python
HOME = {('Australia', 'PLACE'), ('Australian', 'NORP')}     # in about half of all articles
accepted = t['chunk_subtopics'].query("relevance == 'on_topic' and scope in ['australian', 'mixed']")
m = t['entity_mentions']
body = m[~m.in_title & m.chunk_id.isin(accepted.chunk_id)]
body = body[[(e, ty) not in HOME for e, ty in zip(body.entity, body.type)]]
top = body.groupby(['entity', 'type']).article_id.nunique().sort_values(ascending=False).head(20)
```

On the final corpus the top five are Australian Labor Party (58 articles), Sydney (53), Melbourne (46), One Nation (43) and Coalition (41).

### 6.4 Add political lean (Hayden's output)

```python
lean = pd.read_csv('political_lean_predictions.csv')           # article_id, political_label
shown = aus.merge(lean, on='article_id', how='left')           # left / centre / right
```

Hayden's file has 252 rows against 253 articles, so one article (`ART_CBC6CB083FA8`, immigration, The Age) has no lean. Keep the left join and handle the missing label.

### 6.5 Analyse a newly fetched article

Entities work on any article. Subtopics do not (see section 9).

```python
import spacy
from src.topics_entities import extract_entities
from src.topics_entities.config import NER_DISABLE, NER_MODEL

@st.cache_resource                       # load the model once per server, not per request
def get_nlp():
    return spacy.load(NER_MODEL, disable=NER_DISABLE)

mentions, entities = extract_entities(new_articles_df, nlp=get_nlp())
# new_articles_df needs: article_id, topic, title, publisher, clean_text
```

Loading takes about 5 seconds, and each article then takes well under a second. An article with no entities, empty text or a missing title returns empty tables, not an error. The article's `topic` has to come from somewhere (the search query or the collection step). Hayden's `predict_political_leaning(text)` returns `{"label", "confidence"}` for the same article.

### 6.6 Display rules

| Rule | Reason |
|---|---|
| Show `label`, not the subtopic number | Numbers can change on re-runs |
| Hide `Australia` and `Australian` by default | In about half of all articles, so they distinguish nothing |
| Say "articles that mention", never "mentions" | One long article can dominate mention counts |
| Mark `scope = foreign` subtopics, or filter them out of Australian views | The Guardian's main subtopic is foreign for 40% of its articles, against 0–7% for every other outlet |
| Show how many articles sit behind each number | Six subtopics have 6 or fewer articles |

### 6.7 A caution about lean and subtopics

Hayden's lean classifier works from article text embeddings (`all-mpnet-base-v2`, 200-word chunks), and our subtopics are built from passage text embeddings. They are **not independent**, and the joined data shows it:

| Subtopic (main subtopic of the article) | Centre | Left | Right |
|---|---|---|---|
| House prices and the property market | 20 | 0 | 0 |
| Interest rates, inflation and the housing downturn | 9 | 0 | 0 |
| Australian energy policy: data centres, coal and renewables | 2 | 24 | 1 |
| Australian migration policy and politics | 0 | 6 | 25 |
| One Nation and the politics of immigration | 0 | 3 | 13 |

So "left-leaning articles cover subtopic X more" may be partly true by construction, because the label is itself predicted from content like the subtopic. Compare lean **within** one subtopic, show the counts, and avoid presenting subtopic frequency by lean as a finding about outlets. Publisher mix matters too: 47 of The Guardian Australia's 63 articles are labelled left.

## 7. Running, installing and testing

### Install

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf en_core_web_sm
```

The first run downloads the embedding model from Hugging Face (about 130 MB), and `en_core_web_trf` is about 450 MB. If a run hangs right after "Loading weights", the online check is stuck: set `HF_HUB_OFFLINE=1` once the models are downloaded.

### Run

```bash
# from the repository root
python -m src.topics_entities --input data/processed/FINAL_newsprism_corpus.csv --output data/enriched/topics_entities
```

```python
from src.topics_entities import run, write
result = run(corpus_df)                      # five DataFrames, plus result.checks
write(result, 'data/enriched/topics_entities')
```

Options: `--cache-dir <folder>` reuses chunk embeddings between runs, and `--decisions <file>` uses a different set of review decisions. The command prints which subtopics are not reviewed and exits with a non-zero code if an automatic check fails. Generated outputs are in `.gitignore` and are not committed.

### Dependencies that matter

| Pin | Reason |
|---|---|
| `transformers==4.53.2` | `spacy-transformers` 1.4 needs `transformers` 4.x |
| `sentence-transformers==5.7.0` | 6.x needs `transformers` 5 |
| `spacy==3.8.16`, `spacy-transformers==1.4.0`, `torch==2.14.1`, `umap-learn==0.5.12`, `hdbscan==0.8.44`, `bertopic==0.17.4` | Versions the outputs were verified on |

If another component needs `transformers` 5.x, the entity model has to change (see section 9). `scikit-learn` and `pandas` are listed unpinned in the shared requirements; outputs were verified on scikit-learn 1.9.1 and pandas 3.0.6. Hayden's classifier is a pickled scikit-learn model, so keep the scikit-learn version it was trained with in mind when installing.

### Automatic checks on every run

Only the four kept entity types; no outlet mentions of itself; unambiguous surnames resolved; one type per entity; all article and passage IDs join to the input; at least 95% of articles have an entity. All pass on the final corpus (253 of 253 articles have an entity; 14 ambiguous surnames are kept as written).

### Tests and reproducibility

- **45 unit tests** need no models or data and run in under a second: `python -m unittest discover src/topics_entities/tests`.
- A clean installation in a new virtual environment reproduced the same rows and values in all five tables.
- The same outputs came from the team repository's copy of the corpus.
- Cases a fetched article can throw at it (no entities, empty text, missing text or title, an empty batch) return empty tables, not errors.

## 8. Evidence for the choices

Acceptance checks were written before each part was built or scored, so a failed check meant fixing the build, not changing the check. The two places where a rule was extended afterwards are disclosed below.

### 8.1 Subtopic configuration: a 45-run grid

3 chunkings × 3 embedding models × 5 seeds. **Selection rule, written before ranking:** at least 80% of runs give 4–10 subtopics per topic; at least 80% of articles get a subtopic; at most 1 artefact cluster per run; then rank by mean rank over coherence, diversity, stability and coverage, with ties going to the simpler chunking.

| Configuration | NPMI | Diversity | Stability (ARI) | Coverage | Artefacts / run | Result |
|---|---|---|---|---|---|---|
| **100 words + bge-small** | 0.134 | 0.862 | 0.765 | 0.923 | 0.4 | **Selected** |
| Semantic chunks + MiniLM | 0.128 | 0.873 | 0.703 | 0.933 | 0.0 | Passed, more complex chunking |
| 100 words + mpnet | 0.156 | 0.851 | 0.679 | 0.904 | 1.0 | Passed |
| 150 words + MiniLM | 0.141 | 0.879 | 0.776 | 0.926 | 2.2 | Failed: artefacts |
| 150 words + mpnet | 0.131 | 0.878 | 0.784 | 0.941 | 2.6 | Failed: artefacts |

**Disclosure:** the artefact check was added after the first grid run, when 150-word configurations ranked highest while producing 1–6 single-article clusters per run. It encodes a failure found before the grid was designed. The differences among passing configurations are modest (seed-to-seed NPMI varies by about ±0.009), which is why the human review matters more than the choice among them.

### 8.2 Entity model: a hand-annotated gold set

24 passages (8 per topic, spread across publishers, 102 entities), annotated by hand against written guidelines. The selection rule was fixed in advance: higher strict F1 wins; within 0.02 the smaller model wins; below 0.80 the choice is reconsidered; the transformer replaces the choice only if it beats it by more than 0.02 and reaches 0.80.

| Model | Precision | Recall | F1 |
|---|---|---|---|
| `en_core_web_sm` | 0.695 | 0.723 | 0.709 |
| `en_core_web_lg` | 0.752 | 0.782 | 0.767 |
| **`en_core_web_trf` (selected)** | **0.922** | **0.941** | **0.931** |

**Disclosure:** post-processing rules were extended between scoring runs (for example, dropping the portfolio word in "Energy Minister"). The threshold and the selection rule never changed, and each change is recorded.

### 8.3 Benchmark against a large language model

The same gold passages, the same guidelines, the same post-processing and scoring, with the prompt tuned on three passages outside the gold set and frozen first. `gemini-3.8-flash` was answered for 22 of 24 passages (the API quota ran out for 2), so all four systems are scored on those 22 (93 entities).

| System | Precision | Recall | F1 |
|---|---|---|---|
| `gemini-3.8-flash` | 0.989 | 1.000 | 0.995 |
| `en_core_web_trf` | 0.916 | 0.935 | 0.926 |
| `en_core_web_lg` | 0.732 | 0.763 | 0.747 |
| `en_core_web_sm` | 0.677 | 0.699 | 0.688 |

The language model is more accurate, and the local model reaches about 93% of its F1. The local model was kept because it processes the whole corpus in about 100 seconds, costs nothing, gives identical output on every run, and keeps article text on the machine. The free API quota allowed about 20 answers across several attempts.

**Caveats:** the gold set was drafted with LLM assistance and confirmed by a person, which may favour another language model; 93 entities leaves F1 uncertain by several points; one prompt, one model, one run.

### 8.4 Other design evidence

| Decision | Evidence |
|---|---|
| Chunk at all | Median article about 1,038 tokens against a 256–512 token limit |
| About 100 words | TextTiling-style boundaries on sentence embeddings: median idea length 86 words (interquartile range 52–130) |
| Chunks counted equally in the subtopic mix | Chunk length barely varies (CV 0.15); weighting by words changed the main subtopic of 2 of 227 articles |
| Human review at cluster level | Cross-topic passages became outliers at about the same rate as on-topic ones (21% vs 19%), so outlier status does not signal irrelevance |
| Pinned `transformers` 4.x | Re-encoding every passage on the new versions gave identical topic outputs |

## 9. Limitations, known issues and maintenance

### Limitations

| # | Limitation | Effect | What to do |
|---|---|---|---|
| 1 | **Subtopics need a whole corpus.** Clustering learns themes from all articles at once. | A newly fetched article cannot be assigned a subtopic; re-running on a handful of articles gives no clusters, and re-clustering everything would change the labels. | Entities work on new articles today. A nearest-subtopic "assign" step is the next piece of work (section 10). |
| 2 | **One subtopic is a catch-all.** `clean_energy` subtopic 3 ("Mixed: energy projects, AI data centres and Pacific") has no single theme; about 60% of its passages are energy or climate. | It should not be reported as one theme. | The label says "Mixed". |
| 3 | **Two umbrella subtopics may hold several themes:** Australian migration policy (42 articles) and housing supply and planning (30 articles). | Counts for them blur sub-themes. | Not split after seeing the output, because that would tune to the result. |
| 4 | **Impurities inherit the subtopic's tag.** | For example 4 non-Australian articles sit inside the Australian migration subtopic, and one PNG article supplies 32 of 169 passages of the One Nation subtopic. | Recorded in each subtopic's `reason`. |
| 5 | **One subtopic is off-topic.** | 5 articles are tagged off-topic; 3 of them are in no other subtopic, including one genuine energy story. | The reason field says so. |
| 6 | **Small subtopics.** Six have 6 or fewer articles. | Counts based on them are fragile. | Show the article count beside every number. |
| 7 | **Publisher imbalance.** The Guardian is 22 of 25 articles in the UK climate subtopic. | A publisher comparison on it would mostly measure The Guardian. | It is tagged `foreign`, so Australian views exclude it. |
| 8 | **Entity model on a small gold set.** 24 passages, 102 entities. | F1 0.931 is uncertain by several points. | Enlarge the gold set to about 60 passages. |
| 9 | **Entity types can still be wrong for rare names.** | The corpus-wide vote can mistype a name, for example "Bali" came out as a person in the full run and a place when one article was processed alone; "NT" and "Paris" appear as organisations in the top organisations. | Review the top entities before presenting them. |
| 10 | **No entity roles.** | A person is `PERSON`, a party is `ORG`; there is no "politician" or "party member" field. | Optional future table of sourced political roles. |
| 11 | **Lean and subtopics are not independent** (section 6.7). | Subtopic frequency by lean partly reflects the classifier. | Compare within a subtopic. |

### Maintenance

| Situation | What happens | What to do |
|---|---|---|
| The corpus gains or loses articles | The run completes; subtopics whose passage set changed come out `needs_review` or `unreviewed`, with empty tags and a suggested label | Review the flagged subtopics (`review.py`), append the decisions to `review_decisions.csv` |
| A column is renamed upstream | The run stops with `corpus is missing required columns: [...]` | Fix the column name, or ask the data owner |
| A new outlet appears | Its own name is not removed as a self-mention | Add its names to `PUBLISHER_NAMES` in `entities.py` |
| A different embedding or entity model is wanted | Subtopics and labels change | Re-run the grid and the gold-set comparison, then re-review all subtopics |
| Output looks wrong | | Check `result.checks`; run the 45 tests; compare with the verified outputs |

## 10. Next steps

1. **Assign new articles to existing subtopics.** Save each subtopic's centre in embedding space and assign a new passage to the nearest one, or to none when it is too far away. Acceptance check to fix first: re-assigning the existing corpus must reproduce at least 90% of the fitted subtopics for clustered passages. Needed only if the app analyses articles beyond the current 253.
2. **Enlarge the entity gold set** to about 60 passages, with annotations written without model drafts, for narrower confidence intervals and per-type scores.
3. **Remove boilerplate at extraction time** (per-publisher selectors or an article extractor), so every component works on clean text and this component's cleaning finds less to remove.
4. **Refit and re-review periodically.** Stories that did not exist in July–September 2026 will not match any current subtopic.
5. **Optional entity roles.** A small sourced table (politician, party, institution) would let the app group people by party without assigning a lean to any entity.
6. **Confirm dependency pins** with the other components before they add packages that need `transformers` 5.x.
