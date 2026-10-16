"""Human review step (development tool, never called by pipeline.run).

`write_review` records decisions typed into the review sheet in an append-only history and rewrites
the sheet; the history file is what `pipeline.run` reads as its decisions (review_decisions.csv).
"""
import pandas as pd

from .config import HUMAN_COLS
from .topics import merge_review


def update_history(sheet, history_path):
    """Append decisions filled in on the review sheet to the permanent history; latest edit wins.

    Decisions are never dropped, so a label survives any number of runs in which its cluster
    changes (a one-run backup lost them after two runs).
    """
    cols = ['topic', 'subtopic', 'chunk_set_id', 'article_ids'] + HUMAN_COLS
    history = (pd.read_csv(history_path, encoding='utf-8-sig', dtype=str).fillna('')
               if history_path.exists() else pd.DataFrame(columns=cols))
    if sheet is not None:
        filled = sheet.fillna('')
        filled = filled[filled['label'] != '']
        history = pd.concat([history, filled[cols]]).drop_duplicates(['topic', 'chunk_set_id'], keep='last')
    history_path.parent.mkdir(exist_ok=True)
    history[cols].to_csv(history_path, index=False, encoding='utf-8-sig')
    return history


def write_review(descriptions, review_path, history_path):
    """Record new decisions in the history, then rewrite the review sheet (flagged rows first)."""
    sheet = pd.read_csv(review_path, encoding='utf-8-sig', dtype=str) if review_path.exists() else None
    review = merge_review(descriptions, update_history(sheet, history_path))
    review = review.assign(_unflagged=review.flag == '').sort_values(['_unflagged', 'topic', 'subtopic'])
    cols = (['topic', 'subtopic', 'status', 'flag'] + HUMAN_COLS +
            ['suggested_from', 'keywords', 'n_chunks', 'n_articles', 'n_publishers', 'cross_share',
             'central_examples', 'other_examples', 'cross_topic_examples', 'example_articles',
             'chunk_set_id', 'article_ids'])
    review[cols].to_csv(review_path, index=False, encoding='utf-8-sig')  # utf-8-sig: opens cleanly in Excel
    return review
