import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.topics_entities.review import update_history, write_review
from src.topics_entities.topics import (cap_per_article, chunk_fixed, chunk_subtopics, keyword_preprocess, merge_review,
                                    prepare_corpus, subtopic_mix, tidy_keywords)


def sentence(n_words, tag='w'):
    return ' '.join(f'{tag}{i}' for i in range(n_words - 1)) + ' end.'


class ChunkFixedTest(unittest.TestCase):
    def test_no_chunk_exceeds_cap_and_no_word_is_lost(self):
        text = ' '.join(sentence(n) for n in [30, 45, 60, 20, 80, 15, 40, 25])
        chunks = chunk_fixed(text, target=100, cap=190)
        self.assertTrue(all(len(c.split()) <= 190 for c in chunks))
        self.assertEqual(sum(len(c.split()) for c in chunks), len(text.split()))

    def test_chunks_end_on_sentence_boundaries(self):
        text = ' '.join(sentence(n) for n in [40, 40, 40, 40, 40])
        self.assertTrue(all(c.endswith('end.') for c in chunk_fixed(text, target=100, cap=190)))

    def test_sentence_longer_than_cap_is_split_not_truncated(self):
        chunks = chunk_fixed(sentence(450), target=100, cap=190)
        self.assertEqual([len(c.split()) for c in chunks], [190, 190, 70])

    def test_short_tail_merges_only_if_it_fits_under_cap(self):
        fits = chunk_fixed(' '.join(sentence(n) for n in [100, 30]), target=100, cap=190)
        self.assertEqual([len(c.split()) for c in fits], [130])
        too_big = chunk_fixed(' '.join(sentence(n) for n in [180, 30]), target=100, cap=190)
        self.assertEqual([len(c.split()) for c in too_big], [180, 30])


class KeywordTest(unittest.TestCase):
    def test_target_years_kept_other_numbers_dropped(self):
        tokens = keyword_preprocess('In 2024 we set 82% by 2030 and net zero by 2050 for 1,200 homes').split()
        self.assertIn('2030', tokens)
        self.assertIn('2050', tokens)
        for dropped in ('2024', '82%', '1,200'):
            self.assertNotIn(dropped, tokens)

    def test_spelling_variants_map_to_one_form(self):
        self.assertEqual(keyword_preprocess('Datacentres and data centres'), 'data centre and data centre')

    def test_tidy_drops_generic_and_contained_words(self):
        words = ['australia', 'hanson', 'pauline hanson', 'rent', 'rents', 'australian']
        self.assertEqual(tidy_keywords(words), ['pauline hanson', 'rent', 'rents'])


def chunk_frame(rows):
    return pd.DataFrame(rows, columns=['chunk_id', 'article_id', 'topic', 'publisher', 'chunk_idx', 'subtopic'])


class SubtopicMixTest(unittest.TestCase):
    def setUp(self):
        self.chunks = chunk_frame([
            ('a0', 'A', 'housing', 'SMH', 0, 1), ('a1', 'A', 'housing', 'SMH', 1, 2),
            ('a2', 'A', 'housing', 'SMH', 2, 2), ('a3', 'A', 'housing', 'SMH', 3, -1),
            ('b0', 'B', 'housing', 'ABC', 0, 3), ('b1', 'B', 'housing', 'ABC', 1, 4),
            ('c0', 'C', 'housing', 'ABC', 0, -1),
        ])
        self.mix = subtopic_mix(self.chunks).set_index(['article_id', 'subtopic'])

    def test_outliers_excluded_and_weights_sum_to_one(self):
        self.assertAlmostEqual(self.mix.loc[('A', 1), 'weight'], 1 / 3)
        self.assertAlmostEqual(self.mix.loc[('A', 2), 'weight'], 2 / 3)
        self.assertNotIn('C', self.mix.index.get_level_values('article_id'))

    def test_primary_is_most_chunks_and_ties_go_to_first_appearance(self):
        self.assertTrue(self.mix.loc[('A', 2), 'is_primary'])
        self.assertTrue(self.mix.loc[('B', 3), 'is_primary'])
        self.assertFalse(self.mix.loc[('B', 4), 'is_primary'])
        self.assertEqual(self.mix.groupby(level='article_id')['is_primary'].sum().tolist(), [1, 1])


class ChunkSubtopicsTest(unittest.TestCase):
    def test_every_chunk_kept_with_tags_and_outliers_untagged(self):
        chunks = pd.DataFrame({'chunk_id': ['A_001', 'A_000', 'B_000'], 'article_id': ['A', 'A', 'B'],
                               'topic': ['housing'] * 3, 'publisher': ['ABC News'] * 3, 'subtopic': [0, -1, 0]})
        review = pd.DataFrame({'topic': ['housing'], 'subtopic': ['0'], 'label': ['Rents'],
                               'relevance': ['on_topic'], 'scope': ['australian'], 'scope_country': ['']})
        out = chunk_subtopics(chunks, review)
        self.assertEqual(out['chunk_id'].tolist(), ['A_000', 'A_001', 'B_000'])
        self.assertEqual(out['label'].tolist()[1:], ['Rents', 'Rents'])
        self.assertTrue(pd.isna(out['label'].iloc[0]))


class PrepareCorpusTest(unittest.TestCase):
    def test_missing_required_column_stops_with_a_clear_message(self):
        corpus = pd.DataFrame({'article_id': ['A'], 'topic': ['housing'], 'title': ['t'], 'publisher': ['ABC News']})
        with self.assertRaisesRegex(ValueError, 'clean_text'):
            prepare_corpus(corpus)

    def test_input_is_not_modified_and_model_text_added(self):
        corpus = pd.DataFrame({'article_id': ['A'], 'topic': ['housing'], 'title': ['t'], 'publisher': ['ABC News'],
                               'clean_text': ['Rents rose. Subscribe to our newsletter.']})
        out = prepare_corpus(corpus)
        self.assertNotIn('model_text', corpus.columns)
        self.assertEqual(out['model_text'].tolist(), ['Rents rose.'])


class CapPerArticleTest(unittest.TestCase):
    def test_keeps_at_most_cap_evenly_spaced_chunks_including_both_ends(self):
        chunks = chunk_frame([(f'x{i}', 'X', 'housing', 'SMH', i, -1) for i in range(10)]
                             + [('y0', 'Y', 'housing', 'SMH', 0, -1)])
        kept = chunks[cap_per_article(chunks, 4)]
        self.assertEqual(kept[kept.article_id == 'X'].chunk_idx.tolist(), [0, 3, 6, 9])
        self.assertEqual(len(kept[kept.article_id == 'Y']), 1)


def description(chunk_set_id, article_ids, subtopic=0):
    return {
        'topic': 'immigration', 'subtopic': subtopic, 'flag': '', 'keywords': 'ice, detention',
        'n_chunks': 30, 'n_articles': len(article_ids), 'n_publishers': 3, 'cross_share': 0.0,
        'central_examples': '', 'other_examples': '', 'cross_topic_examples': '', 'example_articles': '',
        'chunk_set_id': chunk_set_id, 'article_ids': ';'.join(article_ids),
    }


class ReviewHistoryTest(unittest.TestCase):
    ARTICLES = ['A1', 'A2', 'A3', 'A4', 'A5']

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sheet = Path(self.tmp.name) / 'review.csv'
        self.history = Path(self.tmp.name) / 'history.csv'

    def tearDown(self):
        self.tmp.cleanup()

    def run_review(self, desc):
        return write_review(pd.DataFrame([desc]), self.sheet, self.history).iloc[0]

    def label_sheet(self, label):
        sheet = pd.read_csv(self.sheet, encoding='utf-8-sig', dtype=str)
        sheet.loc[0, ['label', 'relevance', 'scope']] = [label, 'on_topic', 'foreign']
        sheet.to_csv(self.sheet, index=False, encoding='utf-8-sig')

    def test_decision_reused_only_for_identical_chunk_set(self):
        self.run_review(description('cs_1', self.ARTICLES))
        self.label_sheet('US ICE detention')
        row = self.run_review(description('cs_1', self.ARTICLES))
        self.assertEqual((row.status, row.label), ('reviewed', 'US ICE detention'))

    def test_changed_cluster_keeps_suggestion_across_repeated_runs(self):
        # regression: a one-run backup lost the decision on the second run after a change
        self.run_review(description('cs_1', self.ARTICLES))
        self.label_sheet('US ICE detention')
        changed = description('cs_2', self.ARTICLES[:4] + ['A9'])
        for _ in range(3):
            row = self.run_review(changed)
            self.assertEqual(row.status, 'needs_review')
            self.assertIn('US ICE detention', row.suggested_from)
            self.assertEqual(row.label, '')

    def test_low_overlap_gets_no_suggestion(self):
        self.run_review(description('cs_1', self.ARTICLES))
        self.label_sheet('US ICE detention')
        row = self.run_review(description('cs_3', ['A1', 'B1', 'B2', 'B3', 'B4']))  # 1/9 overlap
        self.assertEqual((row.status, row.suggested_from), ('unreviewed', ''))

    def test_latest_edit_wins_in_history(self):
        self.run_review(description('cs_1', self.ARTICLES))
        self.label_sheet('first label')
        self.run_review(description('cs_1', self.ARTICLES))
        self.label_sheet('corrected label')
        self.run_review(description('cs_1', self.ARTICLES))
        history = update_history(None, self.history)
        self.assertEqual(history[history.chunk_set_id == 'cs_1'].label.tolist(), ['corrected label'])


class MergeReviewTest(unittest.TestCase):
    def test_empty_history_leaves_everything_unreviewed(self):
        out = merge_review(pd.DataFrame([description('cs_1', ['A1'])]), pd.DataFrame())
        self.assertEqual(out.iloc[0].status, 'unreviewed')


if __name__ == '__main__':
    unittest.main()
