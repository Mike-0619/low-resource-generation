"""Build sample-level reports and write new result files safely."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


def prediction_details(sample_ids, sources, predictions, references):
    return [
        {"sample_id": sample_id, "src": source, "pred": prediction, "ref": reference}
        for sample_id, source, prediction, reference in zip(
            sample_ids, sources, predictions, references, strict=True,
        )
    ]


def write_report(path, report, *, ensure_ascii=False):
    write_new_file(path, lambda stream: json.dump(report, stream, indent=2, ensure_ascii=ensure_ascii))


def require_new_file(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Output already exists: {path}. Choose a new output path.")


def write_new_file(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            write(stream)
            stream.close()
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"Output already exists: {path}. Choose a new output path.") from exc
        finally:
            temporary.unlink()
