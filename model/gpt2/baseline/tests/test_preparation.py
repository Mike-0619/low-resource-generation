"""English/German preparation, provenance and safe publication."""
import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq

from src.clean_dataset import checksum
from src import clean_dataset, train_tokenizer, utils

from src.utils import PROJECT_ROOT, staged_directory


def row(identifier, category, text, **extra):
    return {'doc-id': identifier, 'category': category, 'language': 'deu',
            'text': text, 'num-tokens': len(text.split()), **extra}


class PreparationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = self.root / 'raw'
        self.raw.mkdir()
        self.output = self.root / 'baseline'
        self.cfg, _ = utils.load_experiment('configs/German/108M.yaml')
        self.cfg['dataset'].pop('download_manifest')
        self.cfg['dataset'].update(corpus_path='raw', sources=['child-books','padding-opensubtitles'])

    def inputs(self, files):
        rows = []
        for filename, records in files.items():
            path = self.raw / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(records), path)
            rows.extend(records)
        self.cfg['dataset'].update(files=list(files), expected_documents=len(rows),
                                   expected_whitespace_tokens=sum(r['num-tokens'] for r in rows))

    def prepare(self, cfg=None):
        with patch.dict(os.environ, CORPUS_ROOT=str(self.root)), contextlib.redirect_stdout(io.StringIO()):
            clean_dataset.prepare(cfg or self.cfg, self.output)

    @property
    def cleaned(self):
        return utils.experiment_paths(self.cfg, self.output).cleaned

    def test_shards_preserve_per_category_order_and_empty_rows(self):
        # Deliberately supply the files in reverse order; official shard traversal is sorted.
        a = [row('a','child-books','<b>Äpfel</b>  und\tÖl !'),
             row('b','padding-opensubtitles','Subtitle credit\nHallo  Welt !')]
        b = [row('c','child-books',''), row('d','padding-opensubtitles','字幕\n'),
             row('e','child-books','Straße bleibt .')]
        first, second = self.cfg['dataset']['files']
        self.inputs({second: b, first: a})
        original = {p: p.read_bytes() for p in self.raw.rglob('*.parquet')}
        self.prepare()
        report = json.loads((self.cleaned / 'preparation.json').read_text())
        expected_texts = {'child-books': 'Äpfel und Öl!\n\nStraße bleibt.',
                          'padding-opensubtitles': 'Hallo Welt!\n'}
        for category, expected in expected_texts.items():
            records = [r for r in a+b if r['category'] == category]
            self.assertEqual((self.cleaned/f'{category}.train.txt').read_text(), expected)
            stats = report['sources'][category]
            self.assertEqual(stats['output_characters'],len(expected))
            self.assertEqual(stats['cleaned_whitespace_tokens'],len(expected.split()))
            self.assertEqual(stats['documents'],len(records))
            self.assertEqual(stats['empty_cleaned_documents'],1)
        self.assertEqual(report['totals']['documents'],5)
        self.assertEqual(report['template_commit'],utils.TEMPLATE_COMMIT)
        self.assertEqual(report['validation'], {'counts_verified': True})
        self.assertEqual(original,{p:p.read_bytes() for p in original})
        # The JSON report is not tokenizer training text.
        backend = Mock()
        backend.train.side_effect = RuntimeError('stop after file selection')
        with patch('src.train_tokenizer.tokenizer_backend',return_value=(backend,Mock())):
            with self.assertRaisesRegex(RuntimeError,'stop after file selection'):
                train_tokenizer.train_tokenizer(self.cfg,self.output)
        selected = backend.train.call_args.args[0]
        self.assertEqual([Path(p).name for p in selected],['child-books.train.txt','padding-opensubtitles.train.txt'])

    def derived_fixture(self):
        self.cfg['dataset']['size']='10M'
        # Only the boundary document is shortened; derived fields remain provenance.
        files={'child-books.train.parquet':[row('a','child-books','Ein\tBuch',
              **{'bytes':8,'original-num-tokens':5,'is-truncated':True})],
               'padding-opensubtitles.train.parquet':[row('b','padding-opensubtitles','Hallo !',**{'bytes':7,'original-num-tokens':2,'is-truncated':False})]}
        self.inputs(files)
        categories={}
        for filename, records in files.items():
            path=self.raw/filename
            categories[records[0]['category']]={'filename':filename,'sha256':checksum(path),
                'size_bytes':path.stat().st_size,'documents':len(records),
                'whitespace_tokens':sum(r['num-tokens'] for r in records)}
        data={'source':{'repo':self.cfg['dataset']['repo_id'],'revision':self.cfg['dataset']['revision']},
              'documents':2,'whitespace_tokens':4,'text_bytes':15,'target_whitespace_tokens':4,'categories':categories,'algorithm':{'name':'category_word_ten_percent_prefix_v1','truncate_final_document':True}}
        (self.raw/'manifest.json').write_text(json.dumps(data))
        self.cfg['dataset']['manifest']='manifest.json'
        return data

    def test_ten_percent_subset_metadata_and_boundary_prefix_are_preserved(self):
        self.derived_fixture()
        before={p:p.read_bytes() for p in self.raw.iterdir()}
        self.prepare()
        self.assertEqual((self.cleaned/'child-books.train.txt').read_text(),'Ein Buch')
        report=json.loads((self.cleaned/'preparation.json').read_text())
        self.assertEqual(report['validation'], {'counts_verified': True})
        self.assertEqual(report['manifest']['sha256'], checksum(Path(report['manifest']['path'])))
        self.assertEqual(report['totals']['raw_whitespace_tokens'],4)
        self.assertEqual(report['totals']['cleaned_whitespace_tokens'],3)
        self.assertEqual(before,{p:p.read_bytes() for p in before})
        with self.assertRaises(FileExistsError): self.prepare()

    def test_manifest_is_provenance_only_and_inputs_are_hashed(self):
        manifest = self.derived_fixture()
        manifest['source']['revision'] = 'historical-record'
        manifest['documents'] = 99
        manifest['categories']['child-books'].update(sha256='not-checked', documents=99)
        (self.raw / 'manifest.json').write_text(json.dumps(manifest))
        self.prepare()
        report = json.loads((self.cleaned / 'preparation.json').read_text())
        self.assertEqual(report['manifest']['source'], manifest['source'])
        self.assertEqual(report['manifest']['algorithm'], manifest['algorithm'])
        self.assertEqual(report['validation'], {'counts_verified': True})
        for entry in report['inputs']:
            self.assertEqual(entry['sha256'], checksum(self.raw / entry['path']))
        for source, entry in report['sources'].items():
            self.assertEqual(entry['output_sha256'], checksum(self.cleaned / f'{source}.train.txt'))

    def test_bad_rows_and_totals_fail_without_publishing(self):
        records=[row('a','child-books','Hallo Welt'),row('b','padding-opensubtitles','Subtitle credit')]
        self.inputs({'data/train.parquet':records})
        for field,value in [('category','unknown'),('language','eng')]:
            with self.subTest(field=field):
                bad=copy.deepcopy(records); bad[0][field]=value
                pq.write_table(pa.Table.from_pylist(bad),self.raw/'data/train.parquet')
                with self.assertRaises(ValueError): self.prepare()
                self.assertFalse(self.cleaned.exists())
        pq.write_table(pa.Table.from_pylist(records),self.raw/'data/train.parquet')
        for field in ('expected_documents','expected_whitespace_tokens'):
            cfg=copy.deepcopy(self.cfg);cfg['dataset'][field]+=1
            with self.assertRaises(ValueError):self.prepare(cfg)
        self.prepare()
        saved=(self.cleaned/'preparation.json').read_bytes()
        other=copy.deepcopy(self.cfg);other['dataset']['size']='10M';other['dataset']['files']=['missing.parquet']
        with self.assertRaises(ValueError):self.prepare(other)
        self.assertEqual((self.cleaned/'preparation.json').read_bytes(),saved)

    def test_category_file_cannot_contain_another_category(self):
        self.inputs({'child-books.train.parquet':[row('a','padding-opensubtitles','Hallo')],
                     'padding-opensubtitles.train.parquet':[row('b','child-books','Buch')]})
        with self.assertRaisesRegex(ValueError,'misplaced category'):self.prepare()

    def test_download_manifest_is_recorded_without_verification(self):
        self.inputs({'data/train.parquet':[row('a','child-books','Buch'),row('b','padding-opensubtitles','Hallo')]})
        path=self.raw/'data/train.parquet'
        support=self.root/'_support';support.mkdir()
        data={'repositories':[{'repo':self.cfg['dataset']['repo_id'],'revision':self.cfg['dataset']['revision']}],
              'files':[{'path':'raw/data/train.parquet','sha256':checksum(path),'size_bytes':path.stat().st_size}]}
        (support/'download_manifest.json').write_text(json.dumps(data))
        self.cfg['dataset']['download_manifest']='_support/download_manifest.json'
        self.prepare()
        report=json.loads((self.cleaned/'preparation.json').read_text())
        self.assertEqual(report['validation'], {'counts_verified': True})
        self.assertEqual(report['manifest']['sha256'], checksum(Path(report['manifest']['path'])))
        self.assertEqual(report['manifest']['repositories'], data['repositories'])



    def test_prepare_preserves_raw_sources_and_refuses_overwrite(self):
        cfg, _ = utils.load_experiment('configs/English/10M.yaml')
        cfg['dataset'].pop('download_manifest')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / cfg['dataset']['corpus_path']
            raw.mkdir(parents=True)
            texts = {name: '<b>Hello</b>  world !\nSubtitle credits\nNext\u00a0 line .\n'
                     for name in cfg['dataset']['sources']}
            for name, text in texts.items():
                (raw / f'{name}.train.txt').write_text(text, encoding='utf-8')
            saved = {p.name: p.read_bytes() for p in raw.iterdir()}
            target = root / 'experiment'
            with patch.dict(os.environ, CORPUS_ROOT=str(root)), contextlib.redirect_stdout(io.StringIO()):
                clean_dataset.prepare(cfg, target)
                with self.assertRaises(FileExistsError):
                    clean_dataset.prepare(cfg, target)
            cleaned = utils.experiment_paths(cfg, target).cleaned
            expected_texts = {
                'html': 'Hello world!\nSubtitle credits\nNext line.\n',
                'plain': '<b>Hello</b> world!\nSubtitle credits\nNext line.\n',
                'subtitle': '<b>Hello</b> world!\nNext line.\n',
            }
            for name in texts:
                mode = 'html' if name in {'simple_wiki', 'gutenberg'} else 'subtitle' if name == 'open_subtitles' else 'plain'
                self.assertEqual((cleaned / f'{name}.train.txt').read_text(), expected_texts[mode])
            self.assertEqual(saved, {p.name: p.read_bytes() for p in raw.iterdir()})
            self.assertFalse(utils.experiment_paths(cfg, target).tokenizer.exists())
            # Unlisted files do not enter cleaning or the report.
            (raw / 'extra.train.txt').write_text('extra')
            other = root / 'other'
            with patch.dict(os.environ, CORPUS_ROOT=str(root)), contextlib.redirect_stdout(io.StringIO()):
                clean_dataset.prepare(cfg, other)
            other_cleaned = utils.experiment_paths(cfg, other).cleaned
            self.assertEqual(sorted(p.name for p in other_cleaned.glob('*.train.txt')), sorted(saved))
            for path in cleaned.glob('*.train.txt'):
                self.assertEqual(path.read_bytes(), (other_cleaned / path.name).read_bytes())
            report = json.loads((other_cleaned / 'preparation.json').read_text())
            self.assertEqual({entry['path'] for entry in report['inputs']}, set(cfg['dataset']['files']))

    def test_staging_preserves_new_artifacts_and_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'output'
            with self.assertRaises(OSError):
                with staged_directory(target) as staging:
                    (staging / 'new').write_text('new')
                    target.mkdir()
                    (target / 'other-process').write_text('preserve')
            self.assertEqual((target / 'other-process').read_text(), 'preserve')
            self.assertFalse(list(root.glob('.output.building-*')))
            link = root / 'link'
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(FileExistsError):
                utils.require_empty(link)

    def test_document_metadata_is_optional_and_counts_use_actual_text(self):
        records = [row('same', 'child-books', 'Hallo Welt'),
                   row('same', 'padding-opensubtitles', 'Guten Tag')]
        self.inputs({'data/train.parquet': records})
        # Only text, category and language are required for processing.
        for record in records:
            record.pop('doc-id')
            record.pop('num-tokens')
        pq.write_table(pa.Table.from_pylist(records), self.raw / 'data/train.parquet')
        self.prepare()
        report = json.loads((self.cleaned / 'preparation.json').read_text())
        self.assertEqual(report['totals']['documents'], 2)
        self.assertEqual(report['totals']['raw_whitespace_tokens'], 4)

    def test_missing_category_is_rejected_even_when_totals_match(self):
        self.inputs({'data/train.parquet': [row('a', 'child-books', 'Hallo Welt')]})
        with self.assertRaisesRegex(ValueError, 'Missing category documents'):
            self.prepare()
        self.assertFalse(self.cleaned.exists())
