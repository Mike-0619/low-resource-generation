import os
import json
import glob
import argparse
import pandas as pd
import re
from pathlib import Path

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    DEFAULT_EVAL_DIR = os.path.join(PROJECT_ROOT, "results", "eval")
    DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "results", "aggregate")

    parser = argparse.ArgumentParser(description="Aggregate Eval results flexibly without Regex.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("-d", "--dir", default=DEFAULT_EVAL_DIR, help="Target directory containing eval JSONs.")
    source.add_argument("--file", help="Aggregate one evaluation report.")
    parser.add_argument("-o", "--output", type=str, default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument("--batch", type=lambda x: (str(x).lower() == 'true'), default=True,
                                            help="True: Tái tạo cây thư mục. False: Chỉ gom các model ở lớp ngoài cùng thành 1 file.")
    args = parser.parse_args()

    if args.file is not None:
        report = os.path.abspath(args.file)
        if not os.path.isfile(report):
            parser.error(f"Evaluation report not found: {report}")
        base_dir = os.path.dirname(report)
        json_files = [report]
    else:
        base_dir = os.path.abspath(args.dir)
        search_pattern = os.path.join(base_dir, "**", "*eval_results.json")
        json_files = glob.glob(search_pattern, recursive=True)
        if not json_files:
            print(f"Error: No '*eval_results.json' files found in {base_dir}")
            return

    results_dir = os.path.abspath(args.output)
    base_name = os.path.basename(base_dir)

    grouped_data = {}
    skipped_count = 0

    for file_path in json_files:
            json_dir = os.path.dirname(file_path)
            rel_path = os.path.relpath(json_dir, base_dir)
            depth = len(rel_path.split(os.sep)) if rel_path != "." else 0

            # Nếu là chế độ lấy file tổng (batch=False), tự động bỏ qua các file nằm quá sâu (như checkpoints)
            if not args.batch and depth > 1:
                    skipped_count += 1
                    continue

            with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                            data = json.load(f)
                    except json.JSONDecodeError as exc:
                            if args.file is not None:
                                    parser.error(f"Could not parse evaluation report {file_path}: {exc}")
                            print(f"Warning: Corrupted JSON skipped -> {file_path}")
                            continue

            # 1. EXTRACT BASE MODEL & FINETUNED DIR
            model_path = data.get("model_path", "Unknown")
            raw_model_name = data.get("raw_model_name", model_path)

            base_model_name = os.path.basename(os.path.normpath(raw_model_name)) if raw_model_name != "Unknown" else "Unknown"
            finetuned_dir = os.path.basename(os.path.normpath(model_path)) if model_path != "Unknown" else "Unknown"

            revision = data.get("revision", "main")
            seed = data.get("seed", 42)

            # 2. DYNAMIC SORTING
            revision_str = str(revision)
            match_num = re.search(r'\d+', revision_str)

            if match_num:
                    rev_num = int(match_num.group())
                    is_text_only = False
            else:
                    rev_num = 0
                    is_text_only = True

            mean_gold_loss = None
            details = data.get("details", [])
            if details:
                    try:
                            gold_losses = [d["losses"][d["gold_index"]] for d in details if "losses" in d and "gold_index" in d]
                            if gold_losses:
                                    mean_gold_loss = sum(gold_losses) / len(gold_losses)
                    except Exception:
                            pass

            git_info = data.get("git", {})
            git_commit = git_info.get("commit", "Unknown")[:7] if git_info.get("commit") else "Unknown"
            git_dirty = " (Dirty)" if git_info.get("dirty") else ""

            row = {
                    "Raw_Model_Name": raw_model_name,
                    "Finetuned_Dir": finetuned_dir,
                    "Revision": revision,
                    "Seed": seed,
                    "Hits@1 (%)": data.get("hits_at_1", 0),
                    "Hits@5 (%)": data.get("hits_at_5", 0),
                    "MRR": data.get("mrr", 0),
                    "Mean_Gold_Loss": round(mean_gold_loss, 4) if mean_gold_loss else None,
                    "Eval_Dtype": data.get("eval_dtype", "Unknown"),
                    "EOS_Token": data.get("eos_token", "Unknown"),
                    "Git_State": f"{git_commit}{git_dirty}",
                    "Total_Evals": data.get("total_evaluated", 0),
                    "Sort_Is_Text": is_text_only,
                    "Sort_Num": rev_num,
                    "Sort_Text": revision_str,
                    "File_Path": file_path,
            }

            # GROUPING LOGIC
            if args.batch:
                    group_dir = base_dir if json_dir == base_dir else os.path.dirname(json_dir)
                    group_key = os.path.relpath(group_dir, base_dir)
            else:
                    group_key = "."

            if group_key not in grouped_data:
                    grouped_data[group_key] = []
            grouped_data[group_key].append(row)

    if not grouped_data:
            print("No valid JSON files found.")
            return

    # ---------------------------------------------------------
    # EXPORT DATA
    # ---------------------------------------------------------
    print(f"\n--- Aggregating Results (Batch Mode: {args.batch}) ---")
    if skipped_count > 0:
            print(f" * Info: Skipped {skipped_count} deeply nested files.")

    for group_key, rows in grouped_data.items():
            df = pd.DataFrame(rows)

            df = df.sort_values(by=["Raw_Model_Name", "Sort_Is_Text", "Sort_Num", "Sort_Text", "Seed"]).reset_index(drop=True)
            df = df.drop(columns=["Sort_Is_Text", "Sort_Num", "Sort_Text"])

            if args.batch:
                    out_dir = os.path.join(results_dir, base_name, group_key) if group_key != "." else os.path.join(results_dir, base_name)
            else:
                    out_dir = os.path.join(results_dir, base_name)

            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, "compiled_results.csv")
            json_path = os.path.join(out_dir, "compiled_results.json")

            df.to_csv(csv_path, index=False)
            df.to_json(json_path, orient="records", indent=4)

            print(f" 📂 Compiled {len(df)} runs into:\n   L_> {csv_path}")

if __name__ == "__main__":
        main()
