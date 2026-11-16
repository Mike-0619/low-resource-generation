"""Run the multi-stage Paraphrase Slurm jobs."""

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import sysconfig
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

from .config import PROMPT_TYPES, parse_args


def stage_command(mode, config, *options):
    return [sys.executable, "-u", "-m", "src.training.pipeline", "--mode", mode,
            "--config", str(config), *options]


def run_stage(mode, config, *options):
    subprocess.run(stage_command(mode, config, *options), check=True)


def run_gpt2(size, config):
    checkpoint = os.environ.get(f"PATH_{size}_BASE")
    if checkpoint and not Path(checkpoint).is_dir():
        raise FileNotFoundError(f"GPT-2 checkpoint not found: {checkpoint}")
    override = ["--model_path", checkpoint] if checkpoint else []
    run_stage("train", config, *override)
    run_stage("parapluie", config)


def run_llama(config):
    for mode in ("llama_benchmark", "parapluie"):
        for prompt in PROMPT_TYPES:
            run_stage(mode, config, "--prompt_type", prompt)


@contextmanager
def background_process(command, **options):
    process = subprocess.Popen(command, **options)
    try:
        yield process
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def wait_for_judge(server, port, log):
    endpoint = f"http://127.0.0.1:{port}/v1/models"
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"llama-server exited during startup. See {log}")
        try:
            with opener.open(endpoint, timeout=5) as response:
                if response.status == 200:
                    return
        except (URLError, OSError):
            pass
        time.sleep(5)
    raise RuntimeError(f"llama-server startup timed out. See {log}")


def run_judge(config, *, log_dir=Path("logs/comparison")):
    args = parse_args(["--mode", "judge", "--config", str(config)])
    model = Path(args.judge_model_path)
    if not model.is_file():
        raise FileNotFoundError(f"LLM Judge model file not found: {model}")

    cuda_lib = Path(sysconfig.get_path("platlib")) / "nvidia/cu13/lib"
    library_path = [str(cuda_lib)]
    if current := os.environ.get("LD_LIBRARY_PATH"):
        library_path.append(current)
    server_env = dict(os.environ, LD_LIBRARY_PATH=os.pathsep.join(library_path))
    subprocess.run(
        [sys.executable, "-c", "import llama_cpp; import sys; sys.exit(0 if "
         "llama_cpp.llama_supports_gpu_offload() else "
         "'The judge requires a CUDA-enabled llama-cpp-python build and an allocated GPU.')"],
        env=server_env,
        check=True,
    )

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    log = log_dir / f"server_{job_id}_{port}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "llama_cpp.server", "--model", str(model),
               "--port", str(port), "--n_gpu_layers", "-1", "--n_ctx", "4096"]
    client_command = stage_command("judge", config, "--port", str(port))
    with log.open("w") as output, background_process(
        command, env=server_env, stdout=output, stderr=subprocess.STDOUT,
    ) as server:
        print(f"Waiting for llama-server on port {port}. Log: {log}", flush=True)
        wait_for_judge(server, port, log)
        with background_process(client_command) as client:
            if status := client.wait():
                raise subprocess.CalledProcessError(status, client_command)


def run_pipeline(args):
    config_dir = args.config_dir
    scope = "without_100m" if args.without_100m else "with_100m"
    if args.part != "judge":
        sizes = ("10M",) if args.without_100m else ("10M", "100M")
        for size in sizes:
            run_gpt2(size, config_dir / "gpt2" / f"gpt2_{size.lower()}.yaml")
        run_llama(config_dir / "llama3/llama3.yaml")
        run_stage("database", config_dir / "comparison" / f"aggregate_{scope}.yaml")
    if args.part != "train":
        run_judge(config_dir / "comparison" / f"judge_{scope}.yaml", log_dir=Path("logs/pipeline"))


def parse_job_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    jobs = parser.add_subparsers(dest="job", required=True)
    for name in ("gpt2", "llama", "judge"):
        job = jobs.add_parser(name, allow_abbrev=False)
        job.add_argument("--config", type=Path, required=True)
        if name == "gpt2":
            job.add_argument("--size", choices=("10M", "100M"), required=True)
    pipeline = jobs.add_parser("pipeline", allow_abbrev=False)
    pipeline.add_argument("--config-dir", type=Path, default=Path("configs"))
    pipeline.add_argument("--without-100m", action="store_true")
    pipeline.add_argument("--train_only", dest="part", action="store_const", const="train")
    pipeline.add_argument("--judge_only", dest="part", action="store_const", const="judge")
    return parser.parse_args(argv)


def main():
    args = parse_job_args()
    if args.job == "gpt2":
        run_gpt2(args.size, args.config)
    elif args.job == "llama":
        run_llama(args.config)
    elif args.job == "judge":
        run_judge(args.config)
    else:
        run_pipeline(args)


def exit_on_term(signum, _frame):
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, exit_on_term)
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except subprocess.CalledProcessError as error:
        status = error.returncode
        sys.exit(status if status > 0 else 128 - status)
