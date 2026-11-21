"""Offline API streaming, completion, concurrency and import isolation."""
import subprocess
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tasks.lambada import src as lambada
from tasks.lambada.src import greedy as word, data as dataset, local, api as backend_api, scoring
MODELS = ("openai/gpt-5.6-luna",)


def event(kind, **fields):
    return SimpleNamespace(type=kind, **fields)


class ResponseStream:
    def __init__(self, events):
        self.events = events
        self.closed = False
        self.read_count = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def __iter__(self):
        for item in self.events:
            self.read_count += 1
            if isinstance(item, Exception):
                raise item
            yield item


def response(text="hello", status="completed", reason=None):
    details = SimpleNamespace(reason=reason) if reason else None
    return ResponseStream([
        event("response.output_text.delta", delta=text),
        event(f"response.{status}", response=SimpleNamespace(incomplete_details=details)),
    ])


@contextmanager
def api(responses):
    client = MagicMock()
    client.closed = False
    client.streams = []
    answers = MagicMock(side_effect=responses)

    def create(**kwargs):
        answer = answers(**kwargs)
        client.streams.append(answer)
        return answer

    def close(*args):
        assert all(stream.closed for stream in client.streams), "Client closed before streams"
        client.closed = True

    client.responses.create.side_effect = create
    factory = MagicMock()
    factory.return_value.__enter__.return_value = client
    factory.return_value.__exit__.side_effect = close
    with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=factory)}):
        yield client


class ApiTests(unittest.TestCase):
    def test_api_path_does_not_import_local_model_dependencies(self):
        script = '''
import sys
sys.path.insert(0, sys.argv[1])
sys.modules['torch'] = None
sys.modules['transformers'] = None
from tasks.lambada.src import greedy as w, data as d, api as a
d.load_lambada_examples = lambda _: ([], {'dataset_examples': 0, 'skipped_examples': []})
a.generate = lambda *args: (item for item in ())
metrics, rows = w.evaluate('fixture/model', 'en', backend='api')
assert metrics['total'] == 0 and metrics['accuracy'] is None and rows == []
'''
        result = subprocess.run([sys.executable, "-B", "-c", script, str(lambada.PROJECT_ROOT)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_api_chunks_and_completion_share_local_scoring(self):
        for text, target, expected in [(' “Straße!” next', 'Strasse', True), ('helen', 'Helen', False),
                                       ("dogs'", 'dogs', False), ('.5', '5', False), ('', 'word', False)]:
            for width in (1, 2, 50):
                stream = ResponseStream([*[event('response.output_text.delta', delta=text[i:i+width])
                                           for i in range(0, len(text), width)], event('response.completed')])
                with self.subTest(text=text, width=width), api([stream]) as client:
                    result = backend_api.request(client, {'context':'context'}, MODELS[0], reasoning_effort='none')
                    self.assertEqual(scoring.score_word(result['prediction'], target, result['completed'])['exact_match'], expected)
                    self.assertTrue(stream.closed)
                    args = client.responses.create.call_args.kwargs
                    self.assertEqual(args['max_output_tokens'], 128)
                    self.assertEqual(args['instructions'], backend_api.API_INSTRUCTIONS)
                    self.assertEqual(args['model'], MODELS[0])
                    self.assertEqual(args['temperature'], 0)
                    self.assertEqual(args['reasoning'], {'effort':'none'})

    def test_api_caps_errors_and_missing_terminal_events(self):
        for text, complete in [('Straße', False), ('Straße ', True)]:
            with api([response(text, 'incomplete', 'max_output_tokens')]) as client:
                result = backend_api.request(client, {'context':'context'}, MODELS[0])
                self.assertEqual(result['completed'], complete)
                self.assertEqual(result['stop_reason'], 'whitespace' if complete else 'token_limit')
        for ending in ([], [event('response.failed')], [event('error')], [OSError('broken stream')]):
            with api([ResponseStream([event('response.output_text.delta', delta='Str'), *ending])]) as client:
                with self.assertRaises((RuntimeError, OSError)):
                    backend_api.request(client, {'context':'context'}, MODELS[0])
                self.assertEqual(client.responses.create.call_count, 1)

    def test_submission_window_handles_empty_small_and_single_worker_inputs(self):
        for concurrency in (1, 3, 8):
            for count in (0, 1, 2, 6):
                consumed = []
                def examples():
                    for index in range(count):
                        consumed.append(index)
                        yield {'context': str(index)}
                with (self.subTest(concurrency=concurrency, count=count),
                      api(lambda **args: response('word' + args['input'] + ' ')) as client):
                    results = backend_api.generate(examples(), MODELS[0], concurrency)
                    for index in range(count):
                        self.assertEqual(next(results)['prediction'], f'word{index}')
                        self.assertEqual(len(consumed), min(count, index + concurrency))
                    with self.assertRaises(StopIteration):
                        next(results)
                    self.assertEqual(client.responses.create.call_count, count)
                    self.assertTrue(client.closed)

    def test_early_close_does_not_submit_more_inputs_and_closes_streams(self):
        for concurrency in (1, 3):
            consumed = []
            def examples():
                for index in range(6):
                    consumed.append(index)
                    yield {'context': str(index)}
            with self.subTest(concurrency=concurrency), api(lambda **args: response('word ')) as client:
                results = backend_api.generate(examples(), MODELS[0], concurrency)
                self.assertEqual(next(results)['prediction'], 'word')
                results.close()
                self.assertEqual(consumed, list(range(concurrency)))
                self.assertLessEqual(client.responses.create.call_count, concurrency)
                self.assertTrue(client.closed)
                self.assertTrue(all(stream.closed for stream in client.streams))

    def test_request_and_input_errors_propagate_and_close_pending_streams(self):
        for failure in ('request', 'initial_input', 'refill_input'):
            consumed = []
            def examples():
                for index in range(6):
                    if index == (1 if failure == 'initial_input' else 3) and failure != 'request':
                        raise OSError('input failed')
                    consumed.append(index)
                    yield {'context': str(index)}
            def answer(**args):
                return ResponseStream([OSError('request failed')]) if failure == 'request' and args['input'] == '0' else response('word ')
            with self.subTest(failure=failure), api(answer) as client:
                results = backend_api.generate(examples(), MODELS[0], 3)
                if failure == 'refill_input':
                    next(results)
                with self.assertRaisesRegex(OSError, 'failed'):
                    next(results)
                self.assertEqual(len(consumed), 1 if failure == 'initial_input' else 3)
                self.assertLessEqual(client.responses.create.call_count, len(consumed))
                self.assertTrue(client.closed)
                self.assertTrue(all(stream.closed for stream in client.streams))

    def test_concurrent_api_results_keep_input_order(self):
        barrier, lock = Barrier(3), Lock()
        active = peak = 0
        def answer(**kwargs):
            index = int(kwargs['input'])
            def events():
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                if index < 3:
                    barrier.wait(timeout=10)
                with lock:
                    active -= 1
                yield event('response.output_text.delta', delta=f'word{index} ')
            return ResponseStream(events())
        with api(answer) as client:
            rows = list(backend_api.generate([{'context':str(i)} for i in range(6)], MODELS[0], 3))
            self.assertEqual([r['prediction'] for r in rows], [f'word{i}' for i in range(6)])
            self.assertEqual(peak, 3)
            self.assertTrue(client.closed)


if __name__ == '__main__':
    unittest.main()
