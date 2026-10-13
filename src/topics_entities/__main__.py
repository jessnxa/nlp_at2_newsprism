"""Command line: python -m src.topics_entities --input corpus.csv --output data/enriched/topics_entities"""
import argparse
import os
import sys

import pandas as pd

from .pipeline import run, write

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')  # silences a harmless fork warning


def main(argv=None):
    p = argparse.ArgumentParser(prog='topics_entities', description=__doc__)
    p.add_argument('--input', required=True, help='corpus CSV (article_id, topic, title, publisher, clean_text)')
    p.add_argument('--output', required=True, help='folder for the output CSV files')
    p.add_argument('--decisions', help='review decisions CSV (default: the packaged decisions)')
    p.add_argument('--cache-dir', help='folder to cache chunk embeddings between runs')
    a = p.parse_args(argv)

    result = run(pd.read_csv(a.input), decisions=a.decisions, cache_dir=a.cache_dir)
    paths = write(result, a.output)
    for name, path in paths.items():
        print(f'{path}')
    print(f'{result.n_unassigned_articles} articles had no subtopic')
    pending = result.subtopics[result.subtopics['status'] != 'reviewed']
    if len(pending):
        print(f'{len(pending)} subtopics are not reviewed (status needs_review / unreviewed):')
        print(pending[['topic', 'subtopic', 'status', 'keywords']].to_string(index=False))
    failed = [name for name, ok in result.checks.items() if not ok]
    for name, ok in result.checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
