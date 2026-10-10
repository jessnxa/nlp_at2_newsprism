import unittest

import pandas as pd

from src.topics_entities.pipeline import MENTION_COLUMNS, extract_entities


class _Tok:
    def __init__(self, text):
        self.text = text


class _Ent:
    def __init__(self, text, label, end):
        self.text, self.label_, self.end = text, label, end


class _Doc:
    def __init__(self):
        self.ents = [_Ent('Anthony Albanese', 'PERSON', 2)]

    def __len__(self):
        return 3

    def __getitem__(self, i):
        return _Tok('said')


class _FakeNlp:
    """Stands in for the spaCy model: one PERSON entity in every text, or none."""

    def __init__(self, with_entities=True):
        self.with_entities = with_entities

    def pipe(self, texts, batch_size=32):
        for _ in texts:
            doc = _Doc()
            if not self.with_entities:
                doc.ents = []
            yield doc


ARTICLE = {'article_id': 'A1', 'topic': 'housing', 'title': 'Albanese on rents', 'publisher': 'ABC News',
           'clean_text': 'Anthony Albanese said rents would rise. The Reserve Bank held rates on Tuesday.'}


class ExtractEntitiesTest(unittest.TestCase):
    def test_returns_run_table_columns_and_counts_body_mentions_only(self):
        mentions, articles = extract_entities(pd.DataFrame([ARTICLE]), nlp=_FakeNlp())
        self.assertEqual(mentions.columns.tolist(), MENTION_COLUMNS)
        row = articles.iloc[0]
        self.assertEqual((row.entity, row.type, row.n_mentions, bool(row.in_title)),
                         ('Anthony Albanese', 'PERSON', 1, True))

    def test_missing_required_column_stops_with_a_clear_message(self):
        with self.assertRaisesRegex(ValueError, 'title'):
            extract_entities(pd.DataFrame([ARTICLE]).drop(columns=['title']), nlp=_FakeNlp())

    def test_input_is_not_modified(self):
        df = pd.DataFrame([ARTICLE])
        extract_entities(df, nlp=_FakeNlp())
        self.assertEqual(df.columns.tolist(), list(ARTICLE))


    def test_fetched_articles_that_yield_nothing_return_empty_tables_not_errors(self):
        cases = {
            'no entities': (pd.DataFrame([ARTICLE]), _FakeNlp(with_entities=False)),
            'empty text': (pd.DataFrame([dict(ARTICLE, clean_text='')]), _FakeNlp(with_entities=False)),
            'missing text and title': (pd.DataFrame([dict(ARTICLE, clean_text=None, title=None)]),
                                       _FakeNlp(with_entities=False)),
            'empty batch': (pd.DataFrame(columns=list(ARTICLE)), _FakeNlp()),
        }
        for name, (df, nlp) in cases.items():
            with self.subTest(name):
                mentions, articles = extract_entities(df, nlp=nlp)
                self.assertEqual(mentions.columns.tolist(), MENTION_COLUMNS)
                self.assertTrue(mentions.empty and articles.empty)
                self.assertIn('entity', articles.columns)


if __name__ == '__main__':
    unittest.main()
