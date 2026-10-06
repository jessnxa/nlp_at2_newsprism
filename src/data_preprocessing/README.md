# Data Collection & Preprocessing

This module contains the data collection and preprocessing pipeline used to build the shared NewsPrism corpus.

The final corpus covers three topics - **immigration, housing, and clean energy** - across nine Australian news publishers between **1 July and 30 September 2026**.

The processed corpus provides the common input for the downstream NewsPrism NLP components.

## Data Collection

News articles were collected from:

- ABC
- SBS
- The Sydney Morning Herald
- The Age
- The Guardian Australia
- 9News
- 7NEWS
- News.com.au
- Sky News Australia

The initial target was approximately 100 final articles per topic rather than enforcing an exact class size.

GDELT was initially tested for article discovery but produced HTTP 429 rate-limit errors when the collection scope was expanded. Google News RSS was therefore used for the final collection process.

A total of **6,703 candidate articles** were identified:

| Topic | Candidates |
| --- | ---: |
| Immigration | 2,716 |
| Housing | 2,784 |
| Clean energy | 1,203 |
| **Total** | **6,703** |

A reproducible sample of 350 candidates per topic (`random_state=42`) was selected for article extraction.

## Preprocessing Pipeline

The preprocessing workflow consists of the following stages:

1. **Article discovery** - `google_news.py`
   - Collects candidate articles using topic-specific keywords, publisher filters and monthly search windows.

2. **URL resolution** - `resolve_google_news_urls.py`
   - Resolves Google News links to their original publisher URLs and validates the resolved publisher domains.

3. **Article extraction** - `extract_articles.py`
   - Extracts full article text from publisher pages using Trafilatura.
   - 955 of 1,050 sampled articles were successfully extracted (~91%).

4. **Cleaning & quality filtering** - `clean_articles.py`
   - Applies light text cleaning while preserving natural article language for downstream NLP.
   - Retains both `raw_text` and `clean_text`.
   - Removes failed, empty or short articles using a minimum threshold of 200 words.

5. **Relevance filtering** - `filter_relevance.py`
   - Uses topic-specific concept matching to reduce weak or incidental topic mentions.
   - Accepts articles where a relevant concept appears in the title or at least two distinct topic concepts appear in the body.

6. **Deduplication** - `deduplicate_articles.py`
   - Identifies duplicate articles using publisher URLs, cleaned article text and normalised titles.

7. **Event grouping** - `group_events.py`
   - Identifies potential same-event articles using publication date and TF-IDF similarity across article titles and text.
   - Candidate cross-publisher matches were manually reviewed before confirmed articles were assigned shared event identifiers.

8. **Article ID generation** - `article_ids.py`
   - Generates deterministic article IDs from normalised URLs using SHA-256.

9. **Corpus construction** - `build_final_corpus.py`
   - Combines accepted articles into the standardised final NewsPrism corpus.

10. **Corpus validation** - `validate_corpus.py`
    - Checks required fields, missing values, duplicate IDs and URLs, topic values, publisher domains and minimum word counts.

## Final Corpus

The final dataset contains **253 articles across 13 columns**:

| Topic | Final Articles |
| --- | ---: |
| Immigration | 65 |
| Housing | 92 |
| Clean energy | 96 |
| **Total** | **253** |

Final validation confirmed:

- 0 missing required values
- 0 duplicate URLs
- 0 duplicate article text
- 0 duplicate article IDs
- 0 articles below 200 words
- 0 invalid or out-of-range publication dates

The final corpus is stored at:

`data/processed/FINAL_newsprism_corpus.csv`

Each article contains a unique `article_id`, allowing downstream NLP outputs produced by different team members to be reliably linked and merged with the shared corpus.

## Configuration

`config.py` contains the preprocessing configuration, including:

- collection date range
- target topics
- search keywords and relevance concepts
- selected publishers and domains
- project data paths
- minimum article word count