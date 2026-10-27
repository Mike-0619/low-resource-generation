"""Fixed cleaning examples for the official baseline rules."""
import unittest
from unittest.mock import patch

from src import clean_dataset
from src.utils import PROJECT_ROOT


class CleanDatasetTest(unittest.TestCase):
    def test_cleaning_rules(self):
        text = '<p>Hello  world !</p>\nSubtitle credit\n字幕\nOndertitels\nUntertitel bleibt\nGrüße\u00a0 , ja。\n'
        cases = [
            (('simple_wiki', 'gutenberg', 'child-wiki', 'child-books', 'educational'),
             'Hello world!\nSubtitle credit\n字幕\nOndertitels\nUntertitel bleibt\nGrüße, ja。\n'),
            (('childes', 'bnc_spoken', 'switchboard', 'child-directed-speech', 'child-available-speech', 'child-news'),
             '<p>Hello world!</p>\nSubtitle credit\n字幕\nOndertitels\nUntertitel bleibt\nGrüße, ja。\n'),
            (('open_subtitles', 'subtitles', 'padding-opensubtitles'),
             '<p>Hello world!</p>\nUntertitel bleibt\nGrüße, ja。\n'),
        ]
        for sources, expected in cases:
            for source in sources:
                with self.subTest(source=source):
                    self.assertEqual(clean_dataset.clean_text(source, text), expected)

    def test_entrypoint_loads_config_and_dispatches(self):
        for language, sizes in [('English', ('10M', '100M')), ('German', ('10M', '108M'))]:
            for size in sizes:
                with self.subTest(language=language, size=size), patch.object(clean_dataset, 'prepare') as run:
                    clean_dataset.main(['--config', f'configs/{language}/{size}.yaml'])
                    run.assert_called_once()
                    cfg, root = run.call_args.args
                    self.assertEqual((cfg['dataset']['language'], cfg['dataset']['size']), (language, size))
                    self.assertEqual(root, PROJECT_ROOT)
