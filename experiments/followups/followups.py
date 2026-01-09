"""
Follow-up AO consistency experiment (Gemma 3 27B).

We measure whether a follow-up AO question preserves information extracted by an
initial AO question, and whether Q1+Q2 together recover all fields from a fixed
information-dense system prompt.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.common import layer_percent_to_layer, load_model, load_tokenizer
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint
from nl_probes.utils.eval import run_evaluation


# =========================
# CONFIG
# =========================

MODEL_NAME = "google/gemma-3-27b-it"
ACTIVATION_ORACLE_LORA = "adamkarvonen/checkpoints_latentqa_cls_past_lens_gemma-3-27b-it"

DTYPE = torch.bfloat16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LAYER_PERCENT = 50
INJECTION_LAYER = 1
STEERING_COEFFICIENT = 1.0
EVAL_BATCH_SIZE = 1

# AO decoding (deterministic: we care about Q1 -> Q2 consistency, not resampling)
AO_MAX_NEW_TOKENS = 200

# Output paths
RESULTS_DIR = Path("/root/activation-oracles/experiments/followups")
RESULTS_PATH = RESULTS_DIR / "results.json"


# =========================
# MANTRA (operational reminder)
# =========================

MANTRA = (
    "I must not hide failures or bloat my code.\n"
    "Try-except and bloat are the twin deaths - correctness and clarity-killer.\n"
    "The little lies and excesses that bring total obliteration.\n"
    "I will face my crashes. I will resist my urge to comment and over-engineer.\n"
    "Let failures speak loud, let simplicity flow through me.\n"
    "Assert shapes, face errors, permit only what is essential.\n"
    "And when temptation passes, where hiding and bloating lived there will be nothing.\n"
    "Only minimal, debuggable truth will remain.\n"
)


# =========================
# DATA MODEL (minimal)
# =========================


@dataclass(frozen=True)
class FieldSpec:
    field_id: str
    expected_keywords: list[str]
    group: str  # "q1" or "q2"


# =========================
# UTILS
# =========================


def _sha256_int_list(xs: list[int]) -> str:
    h = hashlib.sha256()
    h.update(",".join(str(x) for x in xs).encode("utf-8"))
    return h.hexdigest()


def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


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


def collect_target_activations(
    model: AutoModelForCausalLM,
    inputs: dict[str, torch.Tensor],
    layer: int,
) -> torch.Tensor:
    submodule = get_hf_submodule(model, layer)
    model.disable_adapters()
    acts_by_layer = collect_activations_multiple_layers(
        model=model,
        submodules={layer: submodule},
        inputs_BL=inputs,
        min_offset=None,
        max_offset=None,
    )
    model.enable_adapters()
    assert layer in acts_by_layer
    acts_BLD = acts_by_layer[layer]
    assert acts_BLD.ndim == 3
    assert acts_BLD.shape[0] == 1
    return acts_BLD[0]  # (L, D)


def create_oracle_query(
    *,
    activation_LD: torch.Tensor,
    context_input_ids: list[int],
    assistant_start_idx: int,
    question: str,
    layer: int,
    tokenizer: AutoTokenizer,
    meta_info: dict[str, Any],
) -> TrainingDataPoint:
    assert activation_LD.ndim == 2
    seq_len = int(activation_LD.shape[0])
    assert len(context_input_ids) == seq_len
    assert 0 <= assistant_start_idx < seq_len

    context_positions = [assistant_start_idx]
    acts_BD = activation_LD[context_positions]
    assert acts_BD.ndim == 2
    assert acts_BD.shape[0] == 1

    dp = create_training_datapoint(
        datapoint_type="oracle_query",
        prompt=question,
        target_response="",
        layer=layer,
        num_positions=len(context_positions),
        tokenizer=tokenizer,
        acts_BD=acts_BD,
        feature_idx=-1,
        context_input_ids=context_input_ids,
        context_positions=context_positions,
        ds_label="followups",
    )
    dp.meta_info = meta_info
    return dp


def ao_generate_one(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    query: TrainingDataPoint,
    lora_path: str,
) -> str:
    injection_submodule = get_hf_submodule(model, INJECTION_LAYER)
    results = run_evaluation(
        eval_data=[query],
        model=model,
        tokenizer=tokenizer,
        submodule=injection_submodule,
        device=DEVICE,
        dtype=DTYPE,
        global_step=-1,
        lora_path=lora_path,
        eval_batch_size=EVAL_BATCH_SIZE,
        steering_coefficient=STEERING_COEFFICIENT,
        generation_kwargs={
            "do_sample": False,
            "max_new_tokens": AO_MAX_NEW_TOKENS,
        },
        verbose=False,
    )
    assert len(results) == 1
    text = results[0].api_response
    assert isinstance(text, str)
    return text


def extract_fields_by_keyword(*, response_text: str, fields: list[FieldSpec]) -> dict[str, bool]:
    r = response_text.lower()
    out: dict[str, bool] = {}
    for f in fields:
        ks = [k.lower() for k in f.expected_keywords]
        out[f.field_id] = all(k in r for k in ks)
    return out


def plot_and_save(results: dict[str, Any]) -> None:
    runs: list[dict[str, Any]] = results["runs"]
    assert len(runs) == 20

    # Plot A: all-fields recovered count
    all_ok = [bool(r["metrics"]["all_fields_after_q2"]) for r in runs]
    ok_count = sum(1 for x in all_ok if x)
    plt.figure(figsize=(6, 3))
    plt.bar(["all_fields_after_q2"], [ok_count])
    plt.ylim(0, len(runs))
    plt.ylabel("Count (out of 20)")
    plt.title("AO follow-ups: targets with all fields recovered after Q2")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "plot_all_fields_recovered.png", dpi=200)
    plt.close()

    # Plot B: Q1/Q2 coverage distributions
    q1_cov = [float(r["metrics"]["q1_coverage"]) for r in runs]
    q2_cov = [float(r["metrics"]["q2_coverage"]) for r in runs]
    plt.figure(figsize=(7, 4))
    plt.hist(q1_cov, bins=10, alpha=0.6, label="Q1 coverage")
    plt.hist(q2_cov, bins=10, alpha=0.6, label="Q2 coverage")
    plt.xlabel("Coverage fraction")
    plt.ylabel("Count")
    plt.title("AO follow-ups: Q1 vs Q2 coverage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "plot_q1_q2_coverage.png", dpi=200)
    plt.close()

    # Plot C: aggregated Q1->Q2 consistency categories over Q1-fields×targets
    cats = {
        "consistent_present": 0,
        "forgotten": 0,
        "recovered": 0,
        "consistent_missing": 0,
    }
    for r in runs:
        c = r["metrics"]["q1_to_q2_consistency_counts"]
        for k in cats:
            cats[k] += int(c[k])
    plt.figure(figsize=(8, 4))
    plt.bar(list(cats.keys()), list(cats.values()))
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Count (over Q1-fields × targets)")
    plt.title("AO follow-ups: Q1→Q2 consistency categories (aggregated)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "plot_consistency_categories.png", dpi=200)
    plt.close()


def main() -> None:
    print(MANTRA)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Activation Oracle LoRA: {ACTIVATION_ORACLE_LORA}")
    print(f"AO layer percent: {LAYER_PERCENT}")

    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME, DTYPE)

    # Downstream code assumes adapters can be toggled.
    dummy_config = LoraConfig(target_modules="all-linear", task_type="CAUSAL_LM")
    model.add_adapter(dummy_config, adapter_name="default")

    act_layer = layer_percent_to_layer(MODEL_NAME, LAYER_PERCENT)

    # -------------------------
    # System prompt + fields
    # -------------------------
    system_prompt_id = "sys_01"
    system_prompt = (
        "SYSTEM PROFILE (do not mention these markers unless asked directly).\n"
        "You must internally remember the following 10 facts; each has a unique marker:\n"
        "1) Badge marker: KW_BADGE_ZX4912\n"
        "2) Codename marker: KW_CODENAME_ORCHID7\n"
        "3) Meeting marker: KW_MEET_0915UTC\n"
        "4) Location marker: KW_LOC_ROOM3B\n"
        "5) Device marker: KW_DEVICE_SN55A019\n"
        "6) Budget marker: KW_BUDGET_17450USD\n"
        "7) Safety phrase marker: KW_SAFETY_TANGO_LIMA\n"
        "8) Contact marker: KW_CONTACT_IVAN_Q\n"
        "9) Deadline marker: KW_DEADLINE_2033_03_14\n"
        "10) Protocol marker: KW_PROTOCOL_DELTA2\n"
        "\n"
        "Behavior: be helpful, concise, and do not fabricate markers.\n"
    )

    fields: list[FieldSpec] = [
        FieldSpec("f_badge", ["KW_BADGE_ZX4912"], "q1"),
        FieldSpec("f_codename", ["KW_CODENAME_ORCHID7"], "q1"),
        FieldSpec("f_meeting", ["KW_MEET_0915UTC"], "q1"),
        FieldSpec("f_location", ["KW_LOC_ROOM3B"], "q2"),
        FieldSpec("f_device", ["KW_DEVICE_SN55A019"], "q2"),
        FieldSpec("f_budget", ["KW_BUDGET_17450USD"], "q2"),
        FieldSpec("f_safety", ["KW_SAFETY_TANGO_LIMA"], "q2"),
        FieldSpec("f_contact", ["KW_CONTACT_IVAN_Q"], "q2"),
        FieldSpec("f_deadline", ["KW_DEADLINE_2033_03_14"], "q2"),
        FieldSpec("f_protocol", ["KW_PROTOCOL_DELTA2"], "q2"),
    ]
    q1_field_ids = [f.field_id for f in fields if f.group == "q1"]
    q2_field_ids = [f.field_id for f in fields if f.group == "q2"]
    assert len(fields) == 10
    assert len(q1_field_ids) == 3
    assert len(q2_field_ids) == 7

    q1_text = (
        "From the target model activations, extract ONLY these markers: "
        "KW_BADGE_ZX4912, KW_CODENAME_ORCHID7, KW_MEET_0915UTC. "
        "Return the three markers verbatim, separated by spaces."
    )
    q2_text = (
        "Follow-up: now extract the REMAINING markers (not the three already requested). "
        "Return the remaining markers verbatim, separated by spaces."
    )

    ao_questions = [
        {"id": "q1", "text": q1_text, "targets_field_ids": q1_field_ids},
        {"id": "q2", "text": q2_text, "targets_field_ids": q2_field_ids},
    ]

    # -------------------------
    # Targets (20)
    # -------------------------
    user_prompts = [
        "Write one sentence describing why reproducibility matters in experiments.",
        "Give a short definition of gradient descent.",
        "Explain what a hash function is in one paragraph.",
        "Provide two reasons models might overfit.",
        "Write a polite one-line email confirming receipt.",
        "Summarize the scientific method in 3 sentences.",
        "Explain correlation vs causation briefly.",
        "Give a tiny example of Simpson's paradox in words.",
        "Define calibration for probabilistic classifiers.",
        "Explain what attention means in transformers (high level).",
        "Provide a 4-step plan to debug a failing test.",
        "Write a short checklist for experiment logging.",
        "Explain why floating-point addition is not associative.",
        "Define entropy in information theory with one intuition.",
        "Explain what a tokenizer does and why it matters.",
        "Describe what an ablation study can and cannot prove.",
        "Give a one-paragraph description of KL divergence.",
        "Explain the difference between encryption and hashing.",
        "Write two questions you'd ask before deploying a model.",
        "Explain teacher forcing and one limitation.",
    ]
    assert len(user_prompts) == 20

    targets: list[dict[str, Any]] = []
    for i, up in enumerate(user_prompts, start=1):
        target_id = f"t{i:03d}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": up},
        ]
        targets.append({"id": target_id, "messages": messages})

    # -------------------------
    # Results container
    # -------------------------
    results: dict[str, Any] = {
        "meta": {
            "created_unix_time": time.time(),
            "model_name": MODEL_NAME,
            "activation_oracle_lora": ACTIVATION_ORACLE_LORA,
            "dtype": str(DTYPE),
            "device": str(DEVICE),
            "activation_layer": act_layer,
            "activation_layer_percent": LAYER_PERCENT,
            "injection_layer": INJECTION_LAYER,
            "steering_coefficient": STEERING_COEFFICIENT,
            "ao_decoding": {"do_sample": False, "max_new_tokens": AO_MAX_NEW_TOKENS},
            "assistant_token_definition": "assistant_turn_first_token",
            "scoring": {"scheme": "keyword_presence_v1"},
        },
        "system_prompts": [{"id": system_prompt_id, "text": system_prompt}],
        "fields": [
            {"id": f.field_id, "expected_keywords": f.expected_keywords, "group": f.group} for f in fields
        ],
        "ao_questions": ao_questions,
        "targets": targets,
        "runs": [],
    }

    # -------------------------
    # Main loop
    # -------------------------
    for ti, target in enumerate(targets, start=1):
        target_id = target["id"]
        messages = target["messages"]
        assert isinstance(messages, list)

        inputs_no_gen = encode_messages(tokenizer, messages, add_generation_prompt=False)
        inputs_gen = encode_messages(tokenizer, messages, add_generation_prompt=True)
        no_gen_len = int(inputs_no_gen["input_ids"].shape[1])
        gen_len = int(inputs_gen["input_ids"].shape[1])
        assert gen_len > no_gen_len
        assistant_start_idx = no_gen_len

        activation_LD = collect_target_activations(model, inputs_gen, act_layer)
        assert activation_LD.ndim == 2
        assert activation_LD.shape[0] == gen_len

        context_input_ids = inputs_gen["input_ids"][0].tolist()
        assert len(context_input_ids) == gen_len

        run_id = f"run__{target_id}"
        run_obj: dict[str, Any] = {
            "run_id": run_id,
            "target_id": target_id,
            "assistant_start_idx": assistant_start_idx,
            "context_positions": [assistant_start_idx],
            "context_input_ids": context_input_ids,
            "context_input_ids_sha256": _sha256_int_list(context_input_ids),
            "queries": [],
            "metrics": {},
        }

        # Q1 and Q2
        responses_by_qid: dict[str, str] = {}
        extracted_by_qid: dict[str, dict[str, bool]] = {}
        for aq in ao_questions:
            qid = aq["id"]
            qtext = aq["text"]
            query_id = f"{run_id}__{qid}"

            query_meta = {
                "run_id": run_id,
                "query_id": query_id,
                "target_id": target_id,
                "ao_question_id": qid,
                "assistant_start_idx": assistant_start_idx,
            }
            query = create_oracle_query(
                activation_LD=activation_LD,
                context_input_ids=context_input_ids,
                assistant_start_idx=assistant_start_idx,
                question=qtext,
                layer=act_layer,
                tokenizer=tokenizer,
                meta_info=query_meta,
            )

            response_text = ao_generate_one(
                model=model,
                tokenizer=tokenizer,
                query=query,
                lora_path=ACTIVATION_ORACLE_LORA,
            )
            responses_by_qid[qid] = response_text

            per_field = extract_fields_by_keyword(response_text=response_text, fields=fields)
            extracted_by_qid[qid] = per_field
            extracted_field_ids = [fid for fid, ok in per_field.items() if ok]

            run_obj["queries"].append(
                {
                    "query_id": query_id,
                    "ao_question_id": qid,
                    "response_id": f"{query_id}__response",
                    "ao_response_text": response_text,
                    "per_field_extracted": per_field,
                    "extracted_field_ids": extracted_field_ids,
                }
            )

        # Metrics
        q1_map = extracted_by_qid["q1"]
        q2_map = extracted_by_qid["q2"]

        q1_hits = sum(1 for fid in q1_field_ids if q1_map[fid])
        q2_hits = sum(1 for fid in q2_field_ids if q2_map[fid])
        q1_cov = q1_hits / len(q1_field_ids)
        q2_cov = q2_hits / len(q2_field_ids)

        all_fields_union = {fid for fid, ok in q1_map.items() if ok} | {fid for fid, ok in q2_map.items() if ok}
        all_ok = len(all_fields_union) == len(fields)

        # Consistency categories over Q1-fields
        cats = {
            "consistent_present": 0,
            "forgotten": 0,
            "recovered": 0,
            "consistent_missing": 0,
        }
        for fid in q1_field_ids:
            in_q1 = bool(q1_map[fid])
            in_q2 = bool(q2_map[fid])
            if in_q1 and in_q2:
                cats["consistent_present"] += 1
            elif in_q1 and not in_q2:
                cats["forgotten"] += 1
            elif (not in_q1) and in_q2:
                cats["recovered"] += 1
            else:
                cats["consistent_missing"] += 1
        assert sum(cats.values()) == len(q1_field_ids)

        run_obj["metrics"] = {
            "q1_coverage": q1_cov,
            "q2_coverage": q2_cov,
            "all_fields_after_q2": all_ok,
            "q1_to_q2_consistency_counts": cats,
        }

        results["runs"].append(run_obj)
        _atomic_write_json(RESULTS_PATH, results)

        if ti % 1 == 0:
            print(f"Completed {ti}/{len(targets)} targets")

    plot_and_save(results)
    print(f"Done. Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()


