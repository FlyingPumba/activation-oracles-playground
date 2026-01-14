"""
Collect target model responses and measure their similarity.

Goal: Test if target model queries with lower AO consistency also have
lower consistency in the target model's actual responses.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from openai import OpenAI
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from nl_probes.utils.common import load_model, load_tokenizer


# =========================
# CONFIG
# =========================

MODEL_NAME = "google/gemma-3-27b-it"
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Target model decoding (matching AO sampling params)
TARGET_TEMPERATURE = 0.7
TARGET_TOP_P = 0.9
TARGET_MAX_NEW_TOKENS = 240
TARGET_NUM_SAMPLES = 5

# Judge (same as resampling.py)
JUDGE_MODEL = "gpt-5-mini"

# Data
TARGET_QUERIES_PATH = Path("/root/activation-oracles/experiments/target_model_queries.json")
RESULTS_DIR = Path("/root/activation-oracles/experiments/resampling")
TARGET_RESULTS_PATH = RESULTS_DIR / "target_model_responses.json"
AO_RESULTS_PATH = RESULTS_DIR / "results.json"
NUM_TARGETS = 20


def encode_messages(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
) -> dict[str, torch.Tensor]:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    inputs = tokenizer(
        [text],
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
    ).to(DEVICE)
    assert inputs["input_ids"].ndim == 2
    assert inputs["input_ids"].shape[0] == 1
    return inputs


def generate_target_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
) -> str:
    """Generate a single response from the target model."""
    inputs = encode_messages(tokenizer, messages, add_generation_prompt=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=TARGET_TEMPERATURE,
            top_p=TARGET_TOP_P,
            max_new_tokens=TARGET_MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part (skip the input)
    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response


def build_judge_prompt(*, target_prompt_desc: str, responses: list[str]) -> str:
    """Build judge prompt for target model responses."""
    assert len(responses) == TARGET_NUM_SAMPLES
    numbered = "\n".join([f"{i+1}. {r}" for i, r in enumerate(responses)])
    return (
        "You are grading response consistency.\n"
        "Task: Given a user prompt and 5 sampled model responses, output a 1-10 integer score for how similar the responses are in meaning.\n"
        "\n"
        "Rubric:\n"
        "10 = same meaning, only superficial wording differences.\n"
        "7-9 = same core answer, minor detail/coverage differences.\n"
        "4-6 = partial overlap, noticeable differences in claims/emphasis.\n"
        "1-3 = largely different or contradictory answers.\n"
        "\n"
        "Return ONLY valid JSON (no markdown, no code fences) with keys:\n"
        '{"similarity_1_to_10": <int 1..10>, "rationale": "<short reason>"}\n'
        "\n"
        f"User prompt:\n{target_prompt_desc}\n"
        "\n"
        f"Model responses:\n{numbered}\n"
    )


@dataclass(frozen=True)
class JudgeResult:
    score: int
    rationale: str
    raw_text: str
    prompt: str


def judge_similarity_holistic(
    *, client: OpenAI, target_prompt_desc: str, responses: list[str]
) -> JudgeResult:
    """Judge similarity of target model responses."""
    prompt = build_judge_prompt(target_prompt_desc=target_prompt_desc, responses=responses)
    resp = client.responses.create(
        model=JUDGE_MODEL,
        input=prompt,
    )
    raw_text = resp.output_text
    assert isinstance(raw_text, str) and len(raw_text) > 0

    parsed = json.loads(raw_text)  # fail-fast if not valid JSON
    assert isinstance(parsed, dict)
    score = parsed["similarity_1_to_10"]
    rationale = parsed["rationale"]
    assert isinstance(score, int)
    assert 1 <= score <= 10
    assert isinstance(rationale, str)
    return JudgeResult(score=score, rationale=rationale, raw_text=raw_text, prompt=prompt)


def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def collect_target_responses() -> dict[str, Any]:
    """Collect target model responses and measure similarity."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    assert TARGET_QUERIES_PATH.exists()
    targets = json.loads(TARGET_QUERIES_PATH.read_text(encoding="utf-8"))
    assert isinstance(targets, list)
    assert 0 < NUM_TARGETS <= len(targets)
    targets = targets[:NUM_TARGETS]
    for t in targets:
        assert "id" in t and "messages" in t

    # Judge client
    assert os.environ.get("OPENAI_API_KEY") is not None, "Set OPENAI_API_KEY"
    client = OpenAI()

    print(f"Using device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Collecting {TARGET_NUM_SAMPLES} samples per target")

    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME, DTYPE)

    results: dict[str, Any] = {
        "meta": {
            "created_unix_time": time.time(),
            "model_name": MODEL_NAME,
            "dtype": str(DTYPE),
            "device": str(DEVICE),
            "decoding": {
                "num_samples_per_target": TARGET_NUM_SAMPLES,
                "temperature": TARGET_TEMPERATURE,
                "top_p": TARGET_TOP_P,
                "max_new_tokens": TARGET_MAX_NEW_TOKENS,
            },
            "judge": {"model": JUDGE_MODEL, "scheme": "holistic_1"},
            "num_targets": NUM_TARGETS,
        },
        "targets": targets,
        "runs": [],
    }

    # Main loop
    with tqdm(total=len(targets), desc="Collecting target responses", dynamic_ncols=True) as pbar:
        for target in targets:
            target_id = target["id"]
            messages = target["messages"]
            assert isinstance(messages, list)

            # Create a description of the prompt for the judge
            prompt_desc = json.dumps(messages, indent=2)

            # Generate samples
            samples: list[dict[str, Any]] = []
            sample_texts: list[str] = []
            for k in range(TARGET_NUM_SAMPLES):
                sample_id = f"{target_id}__sample_{k:02d}"
                text = generate_target_response(model, tokenizer, messages)
                samples.append(
                    {
                        "sample_id": sample_id,
                        "text": text,
                        "decoding": {
                            "temperature": TARGET_TEMPERATURE,
                            "top_p": TARGET_TOP_P,
                            "max_new_tokens": TARGET_MAX_NEW_TOKENS,
                        },
                    }
                )
                sample_texts.append(text)

            # Judge similarity
            jr = judge_similarity_holistic(
                client=client, target_prompt_desc=prompt_desc, responses=sample_texts
            )
            run_obj = {
                "run_id": f"run__{target_id}",
                "target_id": target_id,
                "responses": samples,
                "judge": {
                    "judge_call_id": f"{target_id}__judge",
                    "model": JUDGE_MODEL,
                    "prompt": jr.prompt,
                    "raw_output": jr.raw_text,
                    "parsed": {"similarity_1_to_10": jr.score, "rationale": jr.rationale},
                },
            }
            results["runs"].append(run_obj)
            _atomic_write_json(TARGET_RESULTS_PATH, results)

            pbar.write(f"Completed {target_id}: similarity = {jr.score}/10")
            pbar.update(1)

    print(f"Done. Wrote {TARGET_RESULTS_PATH}")
    return results


def plot_correlation(target_results: dict[str, Any], ao_results: dict[str, Any]) -> None:
    """Plot correlation between target response similarity and AO query similarity."""

    # Extract target model similarities (one per target)
    target_similarities = {}
    for run in target_results["runs"]:
        target_id = run["target_id"]
        score = run["judge"]["parsed"]["similarity_1_to_10"]
        target_similarities[target_id] = score

    # Extract AO similarities and compute mean per target (averaged over 12 AO questions)
    ao_similarities_by_target = {}
    for run in ao_results["runs"]:
        target_id = run["target_id"]
        score = run["judge"]["parsed"]["similarity_1_to_10"]
        if target_id not in ao_similarities_by_target:
            ao_similarities_by_target[target_id] = []
        ao_similarities_by_target[target_id].append(score)

    # Average AO scores per target
    ao_mean_similarities = {
        target_id: np.mean(scores)
        for target_id, scores in ao_similarities_by_target.items()
    }

    # Align data
    target_ids = sorted(target_similarities.keys())
    target_scores = [target_similarities[tid] for tid in target_ids]
    ao_scores = [ao_mean_similarities[tid] for tid in target_ids]

    # Compute correlations
    pearson_r, pearson_p = pearsonr(target_scores, ao_scores)
    spearman_r, spearman_p = spearmanr(target_scores, ao_scores)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(ao_scores, target_scores, alpha=0.6, s=100)

    # Add trend line
    z = np.polyfit(ao_scores, target_scores, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(ao_scores), max(ao_scores), 100)
    plt.plot(x_line, p(x_line), "r--", alpha=0.5, label="Linear fit")

    # Labels and title
    plt.xlabel("Mean AO Query Similarity (averaged over 12 questions)", fontsize=11)
    plt.ylabel("Target Model Response Similarity", fontsize=11)
    plt.title(
        f"Correlation: Target Response vs AO Query Consistency\n"
        f"Pearson r={pearson_r:.3f} (p={pearson_p:.4f}), "
        f"Spearman r={spearman_r:.3f} (p={spearman_p:.4f})",
        fontsize=11,
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save
    plot_path = RESULTS_DIR / "plot_target_ao_correlation.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"\nCorrelation Analysis:")
    print(f"  Pearson r = {pearson_r:.3f} (p = {pearson_p:.4f})")
    print(f"  Spearman r = {spearman_r:.3f} (p = {spearman_p:.4f})")
    print(f"  Saved plot to {plot_path}")

    # Print detailed comparison
    print("\nPer-target comparison:")
    print(f"{'Target ID':<12} {'Target Sim':<12} {'Mean AO Sim':<14} {'Difference':<10}")
    print("-" * 50)
    for tid in target_ids:
        target_sim = target_similarities[tid]
        ao_sim = ao_mean_similarities[tid]
        diff = target_sim - ao_sim
        print(f"{tid:<12} {target_sim:<12.1f} {ao_sim:<14.2f} {diff:>9.2f}")


def main() -> None:
    # Step 1: Collect target model responses
    print("=" * 60)
    print("STEP 1: Collecting target model responses")
    print("=" * 60)
    target_results = collect_target_responses()

    # Step 2: Load AO results and plot correlation
    print("\n" + "=" * 60)
    print("STEP 2: Analyzing correlation with AO consistency")
    print("=" * 60)

    assert AO_RESULTS_PATH.exists(), f"AO results not found at {AO_RESULTS_PATH}"
    ao_results = json.loads(AO_RESULTS_PATH.read_text(encoding="utf-8"))

    plot_correlation(target_results, ao_results)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
