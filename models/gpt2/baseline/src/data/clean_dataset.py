"""Official source-specific cleaning for text files and ordered Parquet rows."""

import os
import json
from contextlib import ExitStack
from pathlib import Path
import re

from ..corpora.corpus_common import checksum
from ..training.utils import (
    TEMPLATE_COMMIT, experiment_paths, parse_experiment, require_empty, staged_directory,
    write_json_atomic,
)

SOURCE_CLEANING = {
    'simple_wiki': 'html', 'gutenberg': 'html', 'childes': 'plain',
    'bnc_spoken': 'plain', 'switchboard': 'plain', 'open_subtitles': 'subtitle',
    'child-wiki': 'html', 'child-books': 'html', 'educational': 'html',
    'child-directed-speech': 'plain', 'child-available-speech': 'plain',
    'child-news': 'plain', 'subtitles': 'subtitle', 'padding-opensubtitles': 'subtitle',
    'padding-wikipedia': 'identity', 'padding-fineweb-c': 'identity',
    'padding-turmix': 'identity',
}

PARQUET_LANGUAGE_CODES = {'German': 'deu', 'Turkish': 'tur'}
SUBTITLE_CREDIT_LINE = re.compile(r'^.*(?:(?i:subtitle|ondertitel)|字幕).*$\n?', re.MULTILINE)


def clean_text(source, text):
    """Apply the pinned source-specific rules, including whole-line subtitle removal."""
    if source not in SOURCE_CLEANING:
        raise ValueError(f'Unknown source category: {source}')
    mode = SOURCE_CLEANING[source]
    if mode == 'identity':
        return text
    if mode == 'html':
        text = re.sub(r'<[^>]+>', '', text)
    elif mode == 'subtitle':
        text = SUBTITLE_CREDIT_LINE.sub('', text)
    text = re.sub(r'[ \t\u00a0]+', ' ', text)
    return re.sub(r'[ \t\u00a0]([.,;!?。，；！？])', r'\1', text)


def input_path(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Input must be relative to its corpus root: {name}')
    path = root / relative
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Missing or external input: {path}')
    return path


def inspect_inputs(dataset, corpus_root):
    raw = corpus_root / dataset['corpus_path']
    sources, names = dataset['sources'], dataset['files']
    if not sources or len(set(sources)) != len(sources) or set(sources) - SOURCE_CLEANING.keys():
        raise ValueError('Unknown or duplicate source categories')
    if not names or len(set(names)) != len(names):
        raise ValueError('Empty or duplicate input file list')
    if dataset['format'] not in ('text', 'parquet'):
        raise ValueError('Expected dataset.format text or parquet')
    paths = [input_path(raw, name) for name in sorted(names)]
    provenance = {}
    manifest_name = dataset.get('manifest') or dataset.get('download_manifest')
    if manifest_name:
        path = input_path(raw if dataset.get('manifest') else corpus_root, manifest_name)
        metadata = json.loads(path.read_text(encoding='utf-8'))
        provenance = {'path': str(path), 'sha256': checksum(path)}
        # Record provenance without cross-checking the manifest against the inputs.
        for key in ('source', 'algorithm', 'repositories'):
            if key in metadata:
                provenance[key] = metadata[key]
    inputs = [{'path': str(path.relative_to(raw)), 'size_bytes': path.stat().st_size,
               'sha256': checksum(path)} for path in paths]
    return raw, paths, inputs, provenance


def prepare_dataset(dataset, corpus_root, target, template_commit):
    raw, paths, inputs, provenance = inspect_inputs(dataset, corpus_root)
    stats = {source: {'raw_characters': 0, 'cleaned_characters': 0,
                      'raw_whitespace_tokens': 0, 'cleaned_whitespace_tokens': 0}
             for source in sorted(dataset['sources'])}

    def record(source, original, cleaned):
        item = stats[source]
        item['raw_characters'] += len(original)
        item['cleaned_characters'] += len(cleaned)
        raw_words = len(original.split())
        item['raw_whitespace_tokens'] += raw_words
        item['cleaned_whitespace_tokens'] += raw_words if cleaned == original else len(cleaned.split())

    with staged_directory(target) as staging:
        if dataset['format'] == 'text':
            if dataset['language'] != 'English':
                raise ValueError('Text input is configured only for English')
            if {p.name.removesuffix('.train.txt') for p in paths} != set(stats):
                raise ValueError('Text filenames do not match source categories')
            for path in paths:
                source = path.name.removesuffix('.train.txt')
                original = path.read_text(encoding='utf-8')
                cleaned = clean_text(source, original)
                record(source, original, cleaned)
                (staging / path.name).write_text(cleaned, encoding='utf-8')
        else:
            import pyarrow.parquet as pq
            language_code = PARQUET_LANGUAGE_CODES.get(dataset['language'])
            if language_code is None:
                raise ValueError(f"Parquet input is not configured for {dataset['language']}")
            for item in stats.values():
                item.update(documents=0, empty_cleaned_documents=0)
            with ExitStack() as stack:
                streams = {source: stack.enter_context((staging / f'{source}.train.txt').open('w', encoding='utf-8'))
                           for source in stats}
                for path in paths:
                    file_category = path.name.removesuffix('.train.parquet') if path.name.endswith('.train.parquet') else None
                    for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=['category','language','text']):
                        for row in batch.to_pylist():
                            source, original = row['category'], row['text']
                            if source not in stats or (file_category is not None and source != file_category):
                                raise ValueError(f'Unknown or misplaced category: {source}')
                            if row['language'] != language_code:
                                raise ValueError(f'Unexpected language: {row["language"]}')
                            cleaned = clean_text(source, original)
                            item = stats[source]
                            if item['documents']:
                                streams[source].write('\n')
                            streams[source].write(cleaned)
                            record(source, original, cleaned)
                            item['documents'] += 1
                            item['empty_cleaned_documents'] += int(not cleaned.strip())
            words = sum(s['raw_whitespace_tokens'] for s in stats.values())
            documents = sum(s['documents'] for s in stats.values())
            if documents != dataset['expected_documents'] or words != dataset['expected_whitespace_tokens']:
                raise ValueError(f'Unexpected documents/words: {documents}/{words}')
            if any(item['documents'] == 0 for item in stats.values()):
                raise ValueError('Missing category documents')
        for source, item in stats.items():
            path = staging / f'{source}.train.txt'
            item.update(output_bytes=path.stat().st_size, output_sha256=checksum(path))
            item['output_characters'] = item['cleaned_characters'] + max(0, item.get('documents', 1) - 1)
        write_json_atomic(staging / 'preparation.json', {
            'dataset': dataset, 'template_commit': template_commit, 'cleaning_rule_version': 1,
            'input_root': str(raw), 'inputs': inputs, 'manifest': provenance, 'sources': stats,
            'totals': {key: sum(s.get(key, 0) for s in stats.values()) for key in next(iter(stats.values()))
                       if key not in ('output_sha256',)},
            'validation': {'counts_verified': dataset['format'] == 'parquet'},
            'character_count_note': 'cleaned_characters excludes inter-document newlines; output_characters includes them',
        })


def prepare(cfg, root):
    """Read raw corpora without changing them; write independent cleaned text."""
    target = experiment_paths(cfg, root).cleaned
    require_empty(target)
    prepare_dataset(cfg["dataset"], Path(os.environ["CORPUS_ROOT"]).expanduser(), target, TEMPLATE_COMMIT)
    print(f"Prepared independent cleaned corpus: {target}", flush=True)


def main(argv=None):
    cfg, root = parse_experiment('Prepare a baseline training corpus using source-specific rules.', argv)
    prepare(cfg, root)


if __name__ == "__main__":
    main()
