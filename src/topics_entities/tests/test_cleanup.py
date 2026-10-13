import unittest

from src.topics_entities.cleanup import strip_boilerplate

REAL = [
    'The Reserve Bank held the cash rate at 3.6 per cent on Tuesday.',
    'Rents in Sydney rose eight per cent over the year to August.',
    'Hanson told a podcast that her party wanted lower migration.',
]
PROMO = 'Sign up for our Morning Edition newsletter.'


class StripBoilerplateTest(unittest.TestCase):
    def test_sentence_repeated_across_three_articles_is_removed(self):
        out = strip_boilerplate([f'{r} {PROMO}' for r in REAL])
        self.assertEqual(out, REAL)

    def test_sentence_repeated_in_only_two_articles_is_kept_unless_promo(self):
        repeated = 'Coalition leaders met on Monday to discuss the plan.'
        out = strip_boilerplate([f'{REAL[0]} {repeated}', f'{REAL[1]} {repeated}', REAL[2]])
        self.assertIn(repeated, out[0])

    def test_short_promo_sentence_in_one_article_is_removed(self):
        out = strip_boilerplate([f'{REAL[0]} Subscribe to our newsletter for daily updates.'])
        self.assertEqual(out, [REAL[0]])

    def test_cleaning_text_that_is_already_clean_changes_nothing(self):
        once = strip_boilerplate([f'{r} {PROMO}' for r in REAL] + [f'{REAL[0]} Stream free on 7plus'])
        self.assertEqual(strip_boilerplate(once), once)

    def test_real_reporting_mentioning_a_podcast_is_kept(self):
        self.assertEqual(strip_boilerplate([REAL[2]]), [REAL[2]])

    def test_7news_inline_promos_are_removed(self):
        text = (f'{REAL[0]} WATCH THE VIDEO ABOVE: Rate decision explained '
                f'Know the news with the 7NEWS app: Download today {REAL[1]} Stream free on')
        self.assertEqual(strip_boilerplate([text]), [f'{REAL[0]} {REAL[1]}'])


if __name__ == '__main__':
    unittest.main()
