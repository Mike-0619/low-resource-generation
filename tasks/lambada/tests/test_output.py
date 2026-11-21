"""Transactional result publication and recovery."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tasks.lambada.src import greedy as word

class OutputTests(unittest.TestCase):
    def save_old(self, output):
        output.mkdir()
        previous = {"metrics.json": b'{"run":"old"}', "samples.json": b'[{"run":"old"}]'}
        for name, contents in previous.items():
            (output / name).write_bytes(contents)
        return previous

    def contents(self, output):
        return {path.name: path.read_bytes() for path in output.iterdir()}


    def test_cli_and_writer_do_not_create_lock_files(self):
        for entrypoint in ("cli", "writer"):
            with self.subTest(entrypoint=entrypoint), TemporaryDirectory() as directory:
                output = Path(directory) / "result"
                payload = {"limit": None, "status": "complete"}
                word.write_greedy_results(output, payload, [])
                if entrypoint == "cli":
                    with (patch.object(word, "_evaluate", return_value=(payload, [])),
                          redirect_stdout(io.StringIO())):
                        word.main(["--backend", "api", "--model", "fixture/model", "--language", "en",
                                   "--output_dir", str(output)])
                self.assertEqual(list(Path(directory).rglob("*.lock")), [])
                self.assertEqual(json.loads((output / "metrics.json").read_text()), payload)

    def test_preflight_failure_preserves_previous_files(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            previous = self.save_old(output)
            with (patch.object(word, "_evaluate", side_effect=ValueError("preflight failed")),
                  self.assertRaisesRegex(ValueError, "preflight failed")):
                word.main(["--backend", "api", "--model", "fixture/model", "--language", "en",
                           "--output_dir", str(output)])
            self.assertEqual(self.contents(output), previous)

    def test_limited_run_cannot_replace_full_results_before_evaluation(self):
        for metrics in (
            {"limit": None, "sample_selection": {"mode": "all", "limit": None}},
            {"limit": None},
            {"sample_selection": {"mode": "all"}},
        ):
            with self.subTest(metrics=metrics), TemporaryDirectory() as directory:
                output = Path(directory) / "result"
                word.write_greedy_results(output, metrics, [{"run": "full"}])
                previous = self.contents(output)
                with patch.object(word, "_evaluate") as evaluate:
                    with self.assertRaisesRegex(ValueError, "cannot replace full results"):
                        word.main(["--backend", "api", "--model", "fixture/model", "--language", "en",
                                   "--limit", "1", "--output_dir", str(output)])
                    evaluate.assert_not_called()
                with self.assertRaisesRegex(ValueError, "cannot replace full results"):
                    word.write_greedy_results(output, {"limit": 1}, [{"run": "limited"}])
                self.assertEqual(self.contents(output), previous)

    def test_limited_runs_can_replace_limited_results_and_full_can_replace_either(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            for limit in (2, 1, None, None):
                payload = {"limit": limit, "sample_selection": {
                    "mode": "all" if limit is None else "limit", "limit": limit}}
                with (patch.object(word, "_evaluate", return_value=(payload, [{"limit": limit}])) as evaluate,
                      redirect_stdout(io.StringIO())):
                    args = ["--backend", "api", "--model", "fixture/model", "--language", "en",
                            "--output_dir", str(output)]
                    word.main(args + (["--limit", str(limit)] if limit is not None else []))
                evaluate.assert_called_once()
                self.assertEqual(json.loads((output / "metrics.json").read_text()), payload)

    def test_cleanup_failure_after_publication_keeps_run_successful(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            previous = self.save_old(output)
            original = word.shutil.rmtree

            def rmtree(path, *args, **kwargs):
                if Path(path).name == ".result.previous":
                    raise OSError("backup cleanup failed")
                return original(path, *args, **kwargs)

            stderr = io.StringIO()
            payload = {"limit": None, "status": "complete", "run": "new"}
            with (patch.object(word.shutil, "rmtree", rmtree),
                  patch.object(word, "_evaluate", return_value=(payload, [{"run": "new"}])),
                  redirect_stdout(io.StringIO()), redirect_stderr(stderr)):
                word.main(["--backend", "api", "--model", "fixture/model", "--language", "en",
                           "--output_dir", str(output)])
            self.assertIn("Results saved; could not remove backup", stderr.getvalue())
            self.assertEqual(json.loads((output / "metrics.json").read_text()), payload)
            self.assertEqual(json.loads((output / "samples.json").read_text()), [{"run": "new"}])
            backup = output.with_name(".result.previous")
            self.assertEqual(self.contents(backup), previous)
            word.prepare_output_dir(output)
            self.assertEqual(json.loads((output / "metrics.json").read_text()), payload)
            self.assertFalse(backup.exists())

    def test_staged_writes_do_not_follow_existing_result_symlinks(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            output.mkdir()
            external = Path(directory) / "external.json"
            external.write_bytes(b'{"keep":"unchanged"}')
            for name in ("metrics.json", "samples.json"):
                (output / name).symlink_to(external)
            word.write_greedy_results(output, {"run": "new"}, [])
            self.assertEqual(external.read_bytes(), b'{"keep":"unchanged"}')
            self.assertEqual(json.loads((output / "metrics.json").read_text()), {"run": "new"})
            self.assertEqual(json.loads((output / "samples.json").read_text()), [])

    def test_publication_failures_restore_both_old_files(self):
        for phase in ("backup", "publish"):
            with self.subTest(phase=phase), TemporaryDirectory() as directory:
                output = Path(directory) / "result"
                previous = self.save_old(output)
                original = Path.rename

                def rename(path, target):
                    if (phase == "backup" and path == output or
                            phase == "publish" and path.name == ".result.staging"):
                        raise OSError("rename failed")
                    return original(path, target)

                with patch.object(Path, "rename", rename), self.assertRaises(OSError):
                    word.write_greedy_results(output, {"run": "new"}, [{"run": "new"}])
                self.assertEqual(self.contents(output), previous)
                self.assertFalse(output.with_name(".result.previous").exists())
                self.assertFalse(output.with_name(".result.staging").exists())

    def test_failed_rollback_preserves_backup_and_next_run_recovers(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            previous = self.save_old(output)
            original = Path.rename

            def rename(path, target):
                if path.name in {".result.staging", ".result.previous"}:
                    raise OSError("rename failed")
                return original(path, target)

            with (patch.object(Path, "rename", rename),
                  self.assertRaisesRegex(RuntimeError, "backup is preserved")):
                word.write_greedy_results(output, {"run": "new"}, [{"run": "new"}])
            backup = output.with_name(".result.previous")
            self.assertEqual(self.contents(backup), previous)
            word.prepare_output_dir(output)
            self.assertEqual(self.contents(output), previous)
            self.assertFalse(backup.exists())

    def test_interrupted_publication_states_are_recovered(self):
        for phase in ("staging", "backup", "published"):
            with self.subTest(phase=phase), TemporaryDirectory() as directory:
                output = Path(directory) / "result"
                previous = self.save_old(output)
                staging = output.with_name(".result.staging")
                backup = output.with_name(".result.previous")
                staging.mkdir()
                (staging / "metrics.json").write_bytes(b'{"run":"new"}')
                (staging / "samples.json").write_bytes(b'[{"run":"new"}]')
                if phase != "staging":
                    output.rename(backup)
                if phase == "published":
                    staging.rename(output)
                word.prepare_output_dir(output)
                expected = ({"metrics.json": b'{"run":"new"}', "samples.json": b'[{"run":"new"}]'}
                            if phase == "published" else previous)
                self.assertEqual(self.contents(output), expected)
                self.assertFalse(staging.exists())
                self.assertFalse(backup.exists())

    def test_successful_publication_preserves_unrelated_files_and_cleans_sidecars(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            self.save_old(output)
            (output / "notes.txt").write_bytes(b"keep this")
            word.write_greedy_results(output, {"run": "new"}, [{"run": "new"}])
            self.assertEqual(json.loads((output / "metrics.json").read_text()), {"run": "new"})
            self.assertEqual(json.loads((output / "samples.json").read_text()), [{"run": "new"}])
            self.assertEqual((output / "notes.txt").read_bytes(), b"keep this")
            self.assertFalse(output.with_name(".result.lock").exists())
            self.assertFalse(output.with_name(".result.staging").exists())
            self.assertFalse(output.with_name(".result.previous").exists())


if __name__ == "__main__":
    unittest.main()
