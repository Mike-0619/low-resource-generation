import os
import json
import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt

def main():
    task_root = Path(__file__).resolve().parents[2]
    DEFAULT_OUT_FILE = task_root / "results/figures/checkpoint_eval_metrics.png"

    # Set up argument parsing for the folder path
    parser = argparse.ArgumentParser(description="Plot evaluation metrics across model checkpoints.")
    parser.add_argument(
        "-d", "--dir",
        type=str,
        required=True,
        help="Relative or absolute path to the base directory containing eval JSON files."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=DEFAULT_OUT_FILE,
        help="Output absolute or relative path for the plotted figure."
    )
    args = parser.parse_args()

    # Convert to absolute path to ensure it runs correctly from anywhere
    base_dir = os.path.abspath(args.dir)

    # Search for all JSON files ending with eval_results.json recursively
    search_pattern = os.path.join(base_dir, "**", "*eval_results.json")
    json_files = glob.glob(search_pattern, recursive=True)

    if not json_files:
        print(f"Error: No '*eval_results.json' files found in {base_dir}")
        return

    checkpoints_data = []

    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON in {file_path}. Skipping.")
                continue

        # Extract step number for sorting (e.g., from 'rev_step100_seed_42')
        checkpoint = f"{data.get('model_path', '')} {os.path.relpath(file_path, base_dir)}"
        match_step = re.search(r"(?:checkpoint-|step)(\d+)", checkpoint)
        if match_step:
            step_num = int(match_step.group(1))
            cp_name = f"step{step_num}"
        else:
            parent_dir = os.path.basename(os.path.dirname(file_path))
            cp_name = "main" if "main" in parent_dir else parent_dir
            # Pythia 'main' branch corresponds to 143,000 training steps
            step_num = 143000 if cp_name == "main" else 0

        # Calculate Mean NLL Loss for the GOLD candidate
        mean_gold_loss = 0.0
        details = data.get("details", [])
        if details:
            gold_losses = [d["losses"][d["gold_index"]] for d in details]
            mean_gold_loss = sum(gold_losses) / len(gold_losses)

        checkpoints_data.append({
            "name": cp_name,
            "step": step_num,
            "hits_at_1": data.get("hits_at_1", 0),
            "hits_at_5": data.get("hits_at_5", 0),
            "mrr": data.get("mrr", 0),
            "loss": mean_gold_loss
        })

    # Sort checkpoints: integers (steps) first, then strings (like 'main')
    def sort_func(cp):
        key = cp["step"]
        return (0, key) if isinstance(key, int) else (1, key)

    checkpoints_data.sort(key=sort_func)

    # Extract data for plotting
    x_numeric = [cp["step"] if cp["step"] > 0 else 0.5 for cp in checkpoints_data]
    x_labels = [cp["name"] for cp in checkpoints_data]

    y_hits1 = [cp["hits_at_1"] for cp in checkpoints_data]
    y_hits5 = [cp["hits_at_5"] for cp in checkpoints_data]
    y_mrr = [cp["mrr"] for cp in checkpoints_data]
    y_loss = [cp["loss"] for cp in checkpoints_data]

    # Create a 2x2 grid for 4 plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    metrics = [
        (y_hits1, 'Hits@1 (%)', 'b', 'o'),
        (y_hits5, 'Hits@5 (%)', 'g', 's'),
        (y_mrr, 'MRR', 'r', '^'),
        (y_loss, 'Mean Gold NLL Loss (Lower is better)', 'purple', 'D')
    ]

    for i, (y_data, ylabel, color, marker) in enumerate(metrics):
        ax = axes[i]
        ax.plot(x_numeric, y_data, marker=marker, linestyle='-', color=color, linewidth=2, markersize=8)

        # APPLY LOG SCALE TO X-AXIS
        ax.set_xscale('log')

        # Format the X-axis to show the actual checkpoint names instead of scientific notation
        ax.set_xticks(x_numeric)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')

        ax.set_title(f'{ylabel.split(" ")[0]} over Training Time', fontsize=14, fontweight='bold')
        ax.set_xlabel('Checkpoint step (log scale)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, which="both", linestyle='--', alpha=0.5)

    plt.tight_layout()

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {output_path}")

if __name__ == "__main__":
    main()
