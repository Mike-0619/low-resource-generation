"""Score candidates with the local Qwen judge and resume saved scores."""
import json
import os
import re
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

def build_prompt(source, candidate):
    return (
        "Rate the quality of the following candidate paraphrase of the source sentence on a scale of 1 to 10.\n\n"
        "Judge semantic fidelity, factual consistency, fluency, and whether the wording is genuinely "
        "rephrased. Penalize changed names/numbers/facts, added information, omissions, gibberish, and "
        "verbatim copying.\n\n"
        f"Source Sentence: {source}\n"
        f"Candidate Paraphrase: {candidate}\n\n"
        "Return only one line in the exact format: FINAL SCORE: [X]/10, where X is an integer from 1 to 10."
    )

def parse_score(response):
    match = re.search(r"FINAL SCORE:\s*\[?(\d+)\]?/10", response, re.IGNORECASE)
    if match:
        try:
            score = int(match.group(1))
            if 1 <= score <= 10:
                return score
        except ValueError:
            pass

    tail = response[-150:]
    match = re.search(r"(\d+)\s*/\s*10", tail)
    if match:
        try:
            score = int(match.group(1))
            if 1 <= score <= 10:
                return score
        except ValueError:
            pass

    return None

def run_llm_judge(args):
    """
    Modular execution function for the LLM judge ranking evaluation.
    """
    db_path = getattr(args, "db_path", "central_database.json")
    output_path = getattr(args, "output_path", "llm_judge_results.json")
    max_new_tokens = getattr(args, "max_new_tokens", 16)
    limit_samples = getattr(args, "limit_samples", None)
    port = getattr(args, "port", 8000)
    model_name = getattr(args, "model_name", None)

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at {db_path}")

    with open(db_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sample_ids = [item.get("sample_id") for item in data]
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError("Every central database row must have a sample_id")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Central database contains duplicate sample_id values")

    print(f"Loaded {len(data)} samples from {db_path}")

    client = OpenAI(
        base_url=f"http://127.0.0.1:{port}/v1",
        api_key="not-needed-for-local-hpc",
    )
    if model_name:
        server_model = model_name
    else:
        available_models = list(client.models.list().data)
        if not available_models:
            raise RuntimeError("llama-server returned no models from /v1/models")
        server_model = available_models[0].id
    print(f"Using server model ID: {server_model}")

    eval_data = data
    if limit_samples is not None:
        eval_data = data[:limit_samples]
        print(f"Limiting evaluation to the first {limit_samples} samples.")

    judge_model_path = getattr(args, "judge_model_path", None)
    judge_config = {
        "model_id": server_model,
        "model_path": str(Path(judge_model_path).resolve()) if judge_model_path else None,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
    }
    results = [dict(sample, judge_config=judge_config) for sample in data]
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing_by_id = {
                item.get("sample_id"): item
                for item in existing
                if isinstance(item, dict) and item.get("sample_id")
            }
            for i, sample in enumerate(results):
                sample_id = sample.get("sample_id")
                old = existing_by_id.get(sample_id)
                if old is None:
                    continue
                if old.get("source") != sample.get("source") or old.get("reference") != sample.get("reference"):
                    raise ValueError(f"Existing result content changed for sample_id {sample_id}")
                if "judge_config" in old and old["judge_config"] != judge_config:
                    raise ValueError(f"Judge settings changed for {output_path}. Choose a new output path.")
                candidate_fields = sample.keys() - {"judge_config", "judge_scores", "judge_responses"}
                if any(old.get(field) != sample[field] for field in candidate_fields):
                    raise ValueError(f"Existing candidate text changed for sample_id {sample_id}")
                results[i]["judge_scores"] = old.get("judge_scores", {})
                results[i]["judge_responses"] = old.get("judge_responses", {})
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] Could not resume existing results: {exc}")

    for res_item in results:
        res_item.setdefault("judge_scores", {})
        res_item.setdefault("judge_responses", {})

    print("\nPreparing evaluation requests...")

    result_index = {item["sample_id"]: idx for idx, item in enumerate(results)}
    failed_requests = 0
    for sample in tqdm(eval_data, total=len(eval_data), desc="Scoring Samples"):
        sample_id = sample["sample_id"]
        sample_idx = result_index[sample_id]
        source = sample["source"]
        candidates_dict = {"Reference": sample.get("reference", "")}
        for key, val in sample.items():
            if key not in ["sample_id", "source", "reference", "judge_scores", "judge_responses"] and isinstance(val, str):
                candidates_dict[key] = val

        for model_name, candidate_text in candidates_dict.items():
            if not candidate_text:
                continue

            existing_score = results[sample_idx].get("judge_scores", {}).get(model_name)
            if existing_score is not None:
                continue

            prompt = build_prompt(source, candidate_text)

            try:
                response = client.chat.completions.create(
                    model=server_model,
                    messages=[
                        {"role": "system", "content": "You are a strict expert linguistic evaluator."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=max_new_tokens,
                )
                response_text = (response.choices[0].message.content or "").strip()
                score = parse_score(response_text)
                if score is None:
                    failed_requests += 1
                results[sample_idx]["judge_scores"][model_name] = score
                results[sample_idx]["judge_responses"][model_name] = response_text
            except Exception as exc:
                print(f"\n[ERROR] Request failed for sample {sample_idx} ({model_name}): {exc}")
                failed_requests += 1
                results[sample_idx]["judge_scores"][model_name] = None
                results[sample_idx]["judge_responses"][model_name] = f"ERROR: {exc}"

            try:
                tmp_path = output_path + ".partial"
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                with open(tmp_path, "w", encoding="utf-8") as tf:
                    json.dump(results, tf, indent=2, ensure_ascii=False)
                os.replace(tmp_path, output_path)
            except Exception as exc:
                print(f"[WARN] Failed to write intermediate results: {exc}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Report saved to {output_path}")

    model_scores = {}
    for item in results:
        scores = item.get("judge_scores", {})
        for model_name, score in scores.items():
            if score is None:
                continue
            model_scores.setdefault(model_name, []).append(score)

    print("\n=============================================")
    print("           Evaluation Statistics")
    print("=============================================")

    if not model_scores:
        raise RuntimeError("No scores were successfully parsed")

    avg_scores = {model_name: sum(scores_list) / len(scores_list) for model_name, scores_list in model_scores.items()}
    print("\nModel Average Scores (1-10, higher is better):")
    for model_name, avg_score in sorted(avg_scores.items(), key=lambda item: item[1], reverse=True):
        num_samples = len(model_scores[model_name])
        print(f"  {model_name}: {avg_score:.3f} (based on {num_samples} samples)")

    if failed_requests:
        raise RuntimeError(
            f"Judge finished with {failed_requests} failed or unparsed requests; rerun to resume"
        )
