"""Align model prediction reports by sample ID."""
import json
import os
import pandas as pd

from .reports import require_new_file, write_new_file, write_report


def load_report(report_path, name):
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    details = data.get("details", [])
    if not details:
        raise ValueError(f"Report {name} has no details: {report_path}")

    by_id = {}
    order = []
    for position, detail in enumerate(details):
        sample_id = detail.get("sample_id")
        if not sample_id:
            raise ValueError(
                f"Report {name} detail {position} has no sample_id; regenerate stale reports"
            )
        if sample_id in by_id:
            raise ValueError(f"Report {name} contains duplicate sample_id {sample_id}")
        for field in ("src", "ref", "pred"):
            if not isinstance(detail.get(field), str):
                raise ValueError(f"Report {name} sample {sample_id} has invalid {field!r}")
        by_id[sample_id] = detail
        order.append(sample_id)
    return data, by_id, order

def run_build_database(args):
    """
    Modular execution function for combining stage prediction reports into a central table database.
    """
    reports = getattr(args, "reports", [])
    names = getattr(args, "names", [])
    output_json = getattr(args, "output_json", "central_database.json")
    output_csv = getattr(args, "output_csv", "central_database.csv")

    if not reports or not names:
        raise ValueError("reports and names lists must be provided via config or command line.")

    if len(reports) != len(names):
        raise ValueError("Number of reports must match number of names")
    if len(set(names)) != len(names):
        raise ValueError("Report names must be unique")

    if os.path.realpath(output_json) == os.path.realpath(output_csv):
        raise ValueError('JSON and CSV require different output paths')
    require_new_file(output_json)
    require_new_file(output_csv)

    central_dict = {}
    expected_ids = None
    expected_order = None

    # Load all reports
    for report_path, name in zip(reports, names):
        print(f"Loading {name} from {report_path}...")
        if not os.path.exists(report_path):
            raise FileNotFoundError(f"Report file not found: {report_path}")
        data, details_by_id, order = load_report(report_path, name)
        metrics = data.get("metrics", {})
        print(f"  {name} metrics: {metrics}")

        report_ids = set(details_by_id)
        if expected_ids is None:
            expected_ids = report_ids
            expected_order = order
            for sample_id in expected_order:
                detail = details_by_id[sample_id]
                central_dict[sample_id] = {
                    "sample_id": sample_id,
                    "source": detail["src"],
                    "reference": detail["ref"],
                }
        elif report_ids != expected_ids:
            missing = sorted(expected_ids - report_ids)[:10]
            extra = sorted(report_ids - expected_ids)[:10]
            raise ValueError(
                f"Sample mismatch for {name}: expected {len(expected_ids)}, got "
                f"{len(report_ids)}; missing={missing}, extra={extra}"
            )

        for sample_id in expected_order:
            detail = details_by_id[sample_id]
            row = central_dict[sample_id]
            if row["source"] != detail["src"] or row["reference"] != detail["ref"]:
                raise ValueError(f"Source/reference mismatch for sample_id {sample_id} in {name}")
            row[name] = detail["pred"]

    # Convert to list
    central_list = [central_dict[sample_id] for sample_id in expected_order]
    expected_columns = set(names)
    for row in central_list:
        missing_columns = expected_columns - row.keys()
        if missing_columns:
            raise RuntimeError(
                f"Sample {row['sample_id']} is missing predictions: {sorted(missing_columns)}"
            )

    write_report(output_json, central_list)
    frame = pd.DataFrame(central_list)
    write_new_file(output_csv, lambda stream: frame.to_csv(stream, index=False))

    print(f"\nSuccessfully built central database with {len(central_list)} samples.")
    print(f"Saved to {output_json} and {output_csv}")
