"""Case-sensitive first-word scoring and the permitted normalization rules."""
import unittest

from tasks.lambada.src.scoring import normalize_word
from tasks.lambada.src.scoring import score_word


class ScoringTests(unittest.TestCase):
    def test_normalized_matching_preserves_case_and_word_forms(self):
        for raw, expected in (
            ('insurance?"', 'insurance'), ('„Versicherung?“', 'Versicherung'),
            ('\u2003« " insurance ? »\n', 'insurance'),
            ("dogs'", "dogs'"), ("don't", "don't"), ('dogs’', 'dogs’'),
            ('mother-in-law', 'mother-in-law'), ('.5', '.5'), ('3.14', '3.14'),
            ('École', 'École'), ('e\u0301', 'e\u0301'), ('U.S.', 'U.S'),
            ('...', ''), ('"\t»', ''), ('', ''),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(normalize_word(raw), expected)
                self.assertEqual(normalize_word(expected), expected)
        for prediction, target, completed, expected in (
            ('insurance?"', 'insurance', True, True),
            ('Versicherung', '„Versicherung?“', True, True),
            ('helen', 'Helen', True, False), ("dogs'", 'dogs', True, False),
            ("don't", "don't", True, True), ('.5', '5', True, False),
            ('e\u0301', 'é', True, False), ('the', 'insurance', True, False),
            ('...', '...', True, False), ('', '', False, False),
            ('insurance', 'insurance', False, False),
            ('Straße', 'Strasse', True, True), ('Strasse', 'Straße', True, True),
            ('STRAẞE', 'STRASSE', True, True), ('Straße', 'strasse', True, False),
        ):
            with self.subTest(prediction=prediction, target=target):
                result = score_word(prediction, target, completed)
                self.assertEqual(result['exact_match'], expected)
                self.assertEqual(result['normalized_prediction'], normalize_word(prediction))
                self.assertEqual(result['normalized_target_word'], normalize_word(target))


if __name__ == '__main__':
    unittest.main()
