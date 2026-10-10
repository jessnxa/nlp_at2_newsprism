"""Plug-in entry point: `run(corpus)` returns every table; `write(result, out_dir)` saves them.

    from topics_entities import run
    result = run(corpus_df)
    result.article_subtopics, result.chunk_subtopics, result.subtopics,
    result.entity_mentions, result.article_entities

Nothing is read from or written to the working directory, and the input DataFrame is not modified.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import entities as ent
from . import topics as top
from .config import HUMAN_COLS

DECISIONS_PATH = Path(__file__).with_name('review_decisions.csv')
SUBTOPIC_COLUMNS = (['topic', 'subtopic', 'status', 'flag'] + HUMAN_COLS +
                    ['suggested_from', 'keywords', 'n_chunks', 'n_articles', 'n_publishers', 'cross_share'])
MENTION_COLUMNS = ['article_id', 'topic', 'publisher', 'chunk_id', 'in_title', 'surface', 'entity', 'type']


@dataclass
class Result:
    article_subtopics: pd.DataFrame   # one row per (article, subtopic): weight, is_primary, review tags
    chunk_subtopics: pd.DataFrame     # one row per passage: subtopic (-1 = outlier) and review tags
    subtopics: pd.DataFrame           # one row per subtopic: label, tags, status, keywords, sizes
    entity_mentions: pd.DataFrame     # one row per mention, with chunk_id
    article_entities: pd.DataFrame    # one row per (article, entity)
    checks: dict                      # automatic sanity checks {name: passed}
    n_unassigned_articles: int        # articles with no subtopic (all chunks outliers)


def load_decisions(decisions=None):
    """Review decisions as a DataFrame: None = the packaged decisions, or a path / DataFrame."""
    if decisions is None:
        decisions = DECISIONS_PATH
    if isinstance(decisions, pd.DataFrame):
        return decisions
    return pd.read_csv(decisions, encoding='utf-8-sig', dtype=str).fillna('')


def run(corpus, decisions=None, cache_dir=None, nlp=None):
    """Topics and entities for a corpus DataFrame (needs article_id, topic, title, publisher, clean_text).

    decisions: review decisions (path or DataFrame); default is the packaged review_decisions.csv.
    cache_dir: optional folder to cache chunk embeddings between runs.
    nlp: optional preloaded spaCy pipeline (for tests or repeated runs).
    A subtopic whose chunks differ from the reviewed ones comes out with status 'needs_review' or
    'unreviewed' and empty tags; it never stops the run.
    """
    corpus = top.prepare_corpus(corpus)
    chunks = top.build_chunks(corpus)
    emb = top.cached_embeddings(chunks, cache_dir)
    chunks = chunks.join(top.topic_similarity(chunks, emb))
    chunks['subtopic'], _, diags = top.run_clustering(chunks, emb)

    review = top.merge_review(top.describe_subtopics(chunks, emb, diags), load_decisions(decisions))
    review = review.sort_values(['topic', 'subtopic']).reset_index(drop=True)
    mix, unassigned = top.article_subtopics(chunks, review)

    mentions, ambiguous, ent_chunks = ent.extract_mentions(corpus, nlp)
    articles = ent.article_entities(mentions)
    return Result(
        article_subtopics=mix.reset_index(drop=True),
        chunk_subtopics=top.chunk_subtopics(chunks, review),
        subtopics=review[SUBTOPIC_COLUMNS],
        entity_mentions=mentions[MENTION_COLUMNS],
        article_entities=articles.reset_index(drop=True),
        checks=ent.check_entities(mentions, articles, ent_chunks, corpus, ambiguous),
        n_unassigned_articles=unassigned,
    )


def write(result, out_dir):
    """Save the five tables as UTF-8 CSV files (BOM, so Excel opens them cleanly) in out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {'article_subtopics': result.article_subtopics, 'chunk_subtopics': result.chunk_subtopics,
             'subtopics': result.subtopics, 'entity_mentions': result.entity_mentions,
             'article_entities': result.article_entities}
    for name, df in files.items():
        df.to_csv(out_dir / f'{name}.csv', index=False, encoding='utf-8-sig')
    return {name: out_dir / f'{name}.csv' for name in files}
