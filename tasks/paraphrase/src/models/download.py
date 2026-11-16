import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


def get_args():
    parser = argparse.ArgumentParser(description="Download a model repository from the Hugging Face Hub")
    default_model_dir = Path(__file__).resolve().parents[2] / "models"
    parser.add_argument("--repo_id", type=str, default="unsloth/Qwen3.5-122B-A10B-GGUF")
    parser.add_argument("--local_dir", type=str, default=str(default_model_dir))
    parser.add_argument("--max_workers", type=int, default=1, help="Number of parallel download workers")
    parser.add_argument(
        "--allow_pattern",
        default="UD-IQ4_NL/Qwen3.5-122B-A10B-UD-IQ4_NL-*.gguf",
        help="Only download the three GGUF shards used by scripts/pipeline.slurm and the standalone judge jobs",
    )
    return parser.parse_args()


def main():
    args = get_args()
    print(f"Downloading {args.repo_id} to {args.local_dir} with {args.max_workers} workers...")
    try:
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=args.local_dir,
            max_workers=args.max_workers,
            allow_patterns=[args.allow_pattern],
        )
        print("Download completed successfully.")
    except Exception as exc:
        print(f"Download failed: {exc}")
        raise


if __name__ == "__main__":
    main()
