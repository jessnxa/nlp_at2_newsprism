import unittest

import pandas as pd

from src.topics_entities.entities import (apply_context_rules, clean_mentions, clean_title, is_self_mention, normalise_name,
                                      resolve_org_short_forms, resolve_surnames, singular_group, vote_types)


class NormaliseNameTest(unittest.TestCase):
    def test_possessives_and_quotes(self):
        self.assertEqual(normalise_name('Pauline Hanson’s'), 'Pauline Hanson')
        self.assertEqual(normalise_name("“Labor's"), 'Labor')

    def test_leading_the_dropped_except_real_names(self):
        self.assertEqual(normalise_name('the Reserve Bank'), 'Reserve Bank')
        self.assertEqual(normalise_name('The Age'), 'The Age')
        self.assertEqual(normalise_name('the Hague'), 'the Hague')

    def test_honorific_only_stripped_when_attached_to_a_name(self):
        self.assertEqual(normalise_name('Ms Conisbee'), 'Conisbee')
        self.assertEqual(normalise_name('Prime Minister Anthony Albanese'), 'Anthony Albanese')
        self.assertEqual(normalise_name('Prime Minister'), 'Prime Minister')


class SelfMentionAndTitleTest(unittest.TestCase):
    def test_own_outlet_names_and_sub_brands(self):
        self.assertTrue(is_self_mention('Guardian', 'The Guardian Australia'))
        self.assertTrue(is_self_mention('SBS News Podcasts', 'SBS News'))
        self.assertTrue(is_self_mention('ABC Radio Darwin', 'ABC News'))
        self.assertTrue(is_self_mention('Herald', 'Sydney Morning Herald'))
        self.assertFalse(is_self_mention('Guardian', 'ABC News'))

    def test_outlet_names_that_are_ordinary_words_need_an_exact_match(self):
        self.assertTrue(is_self_mention('The Age', 'The Age'))
        self.assertFalse(is_self_mention('Age Pension', 'The Age'))

    def test_site_suffix_removed_from_titles(self):
        self.assertEqual(clean_title('Rents rise again - SMH.com.au'), 'Rents rise again')
        self.assertEqual(clean_title('UK asylum plan | Immigration | The Guardian'), 'UK asylum plan')
        self.assertEqual(clean_title('Labor - Greens deal on housing'), 'Labor - Greens deal on housing')


def mentions(rows):
    return pd.DataFrame(rows, columns=['article_id', 'entity', 'type'])


class ResolveSurnamesTest(unittest.TestCase):
    def test_resolves_only_with_one_full_name_in_the_same_article(self):
        m = mentions([('A', 'Pauline Hanson', 'PERSON'), ('A', 'Hanson', 'PERSON'), ('B', 'Hanson', 'PERSON')])
        resolved, _ = resolve_surnames(m)
        self.assertEqual(resolved.tolist(), ['Pauline Hanson', 'Pauline Hanson', 'Hanson'])

    def test_ambiguous_surname_kept_and_reported(self):
        m = mentions([('A', 'John Kennedy', 'PERSON'), ('A', 'Mary Kennedy', 'PERSON'), ('A', 'Kennedy', 'PERSON')])
        resolved, ambiguous = resolve_surnames(m)
        self.assertEqual(resolved.iloc[2], 'Kennedy')
        self.assertEqual(ambiguous, [('A', 'Kennedy', ['John Kennedy', 'Mary Kennedy'])])

    def test_hyphenated_surname_never_merges(self):
        m = mentions([('A', 'Sarah Hanson-Young', 'PERSON'), ('A', 'Hanson', 'PERSON')])
        self.assertEqual(resolve_surnames(m)[0].iloc[1], 'Hanson')


class VoteTypesTest(unittest.TestCase):
    def test_one_vote_per_article_so_a_long_article_cannot_dominate(self):
        m = mentions([('A', 'X', 'ORG')] * 10 + [('B', 'X', 'PERSON'), ('C', 'X', 'PERSON')])
        self.assertEqual(set(vote_types(m)), {'PERSON'})

    def test_verified_override_applies(self):
        m = mentions([('A', 'Darwin', 'PERSON'), ('B', 'Darwin', 'PERSON')])
        self.assertEqual(set(vote_types(m)), {'PLACE'})


class CleanMentionsTest(unittest.TestCase):
    def test_generic_institutions_dropped_named_ones_kept(self):
        m = pd.DataFrame({'surface': ['the Federal Government', 'Cabinet', 'the Commonwealth', 'NOM', 'Treasury', 'the RBA'],
                          'publisher': ['ABC News'] * 6, 'type': ['ORG'] * 6})
        self.assertEqual(clean_mentions(m)['entity'].tolist(), ['Treasury', 'Reserve Bank of Australia'])


class SingularGroupTest(unittest.TestCase):
    def test_plural_groups_merge_into_singular(self):
        self.assertEqual(singular_group('Australians', 'NORP'), 'Australian')
        self.assertEqual(singular_group('Kiwis', 'NORP'), 'Kiwi')

    def test_exceptions_unchanged(self):
        self.assertEqual(singular_group('Swiss', 'NORP'), 'Swiss')
        self.assertEqual(singular_group('Greens', 'NORP'), 'Greens')
        self.assertEqual(singular_group('Wales', 'PLACE'), 'Wales')


class ContextRulesTest(unittest.TestCase):
    def test_portfolio_word_in_job_title_dropped(self):
        m = pd.DataFrame({'entity': ['Energy', 'Energy', 'Chris Bowen'], 'type': ['ORG', 'ORG', 'PERSON'],
                          'next_word': ['Minister', 'Secretary', 'said']})
        self.assertEqual(apply_context_rules(m)['entity'].tolist(), ['Chris Bowen'])

    def test_leader_government_becomes_org_only_for_known_surnames(self):
        m = pd.DataFrame({'entity': ['Anthony Albanese', 'Albanese', 'Victorian', 'Donald Trump', 'Trump'],
                          'type': ['PERSON', 'NORP', 'NORP', 'PERSON', 'PERSON'],
                          'next_word': ['said', 'government', 'government', 'said', 'administration']})
        out = apply_context_rules(m)
        self.assertEqual(list(zip(out['entity'], out['type'])),
                         [('Anthony Albanese', 'PERSON'), ('Albanese government', 'ORG'), ('Victorian', 'NORP'),
                          ('Donald Trump', 'PERSON'), ('Trump administration', 'ORG')])


class ResolveOrgShortFormsTest(unittest.TestCase):
    def test_short_form_resolved_only_with_one_longer_name(self):
        m = mentions([('A', 'Oakes Energy Services', 'ORG'), ('A', 'Oakes', 'ORG'),
                      ('B', 'Westpac Group', 'ORG'), ('B', 'Westpac Institute', 'ORG'), ('B', 'Westpac', 'ORG')])
        self.assertEqual(resolve_org_short_forms(m).tolist(),
                         ['Oakes Energy Services', 'Oakes Energy Services', 'Westpac Group', 'Westpac Institute', 'Westpac'])


if __name__ == '__main__':
    unittest.main()
