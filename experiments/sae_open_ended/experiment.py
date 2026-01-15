"""
SAE Open-Ended Experiment: Comparing SAE Top Latents with AO Open-Ended Responses.

Goal: Test if the Activation Oracle's open-ended description of model activations aligns
with the concepts represented by the top-activating SAE latents.

Method:
1. Run 100 diverse user queries through Gemma-3-27B-it to generate responses
2. At the <start_of_turn> token for the assistant, collect activations (layer 31)
3. Use SAE to find top 5 activating latents at that token position
4. Fetch auto-interp descriptions for those latents from Neuronpedia
5. Query the Activation Oracle with "What's the model thinking about?"
6. Use an LLM judge to check if the AO response mentions each latent concept
7. Calculate recall scores and analyze the alignment
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import torch
from openai import OpenAI
from peft import LoraConfig
from sae_lens import SAE
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from nl_probes.utils.activation_utils import (
    collect_activations_multiple_layers,
    get_hf_submodule,
)
from nl_probes.utils.common import layer_percent_to_layer, load_model, load_tokenizer
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint
from nl_probes.utils.eval import run_evaluation
import nl_probes.utils.eval as eval_mod


# =========================
# CONFIG
# =========================

MODEL_NAME = "google/gemma-3-27b-it"
ACTIVATION_ORACLE_LORA = "adamkarvonen/checkpoints_latentqa_cls_past_lens_gemma-3-27b-it"

# SAE Configuration for Gemma-3-27B-it layer 31
SAE_RELEASE = "gemma-scope-2-27b-it-res"  # Residual post SAEs
SAE_ID = "layer_31_width_16k_l0_medium"
NEURONPEDIA_MODEL_ID = "gemma-3-27b-it"  # Model ID for Neuronpedia API
NEURONPEDIA_SAE_ID = "31-gemmascope-2-res-16k"  # SAE ID for Neuronpedia API

DTYPE = torch.bfloat16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LAYER_PERCENT = 50  # 50% depth = layer 31 for Gemma-3-27B
INJECTION_LAYER = 1
STEERING_COEFFICIENT = 1.0
EVAL_BATCH_SIZE = 1

# Target model generation
TARGET_TEMPERATURE = 0.7
TARGET_TOP_P = 0.9
TARGET_MAX_NEW_TOKENS = 100  # Only need start_of_turn activations, keep response short

# AO decoding
AO_TEMPERATURE = 0.7
AO_TOP_P = 0.9
AO_MAX_NEW_TOKENS = 200

# SAE settings
TOP_K_LATENTS = 5

# Known structural/positional latents to always ignore.
# These fire frequently at <start_of_turn> regardless of query content, encoding
# "assistant is about to respond" rather than semantic content. Their auto-interp
# descriptions are misleading or don't match the top activating examples.
# See REPORT.md for analysis.
IGNORED_STRUCTURAL_LATENTS = frozenset({
    512,   # "start of turn questions" - fires on 94% of queries, purely positional
    57,    # "'t / <start_of_turn>" - fires on 89% of queries, tokenization artifact
    102,   # "multilingual technical terms" - fires on 35% of queries, unrelated to content
    114,   # "Russian employment context" - fires on 25% of queries, unrelated to content
    1895,  # Structural latent with misleading auto-interp
    3870,  # "lucid dreams" - fires spuriously, doesn't match activating examples
    11672, # "company name" - fires spuriously, doesn't match activating examples
    543,   # "internet addresses" - fires spuriously, doesn't match activating examples
})

# Judge model
JUDGE_MODEL = "gpt-4o-mini"

# Paths
RESULTS_DIR = Path("/root/activation-oracles/experiments/sae_open_ended")
RESULTS_PATH = RESULTS_DIR / "results.json"

# AO Query for open-ended description
AO_QUERY = "What's the model thinking about?"

# 100 diverse target model queries
TARGET_QUERIES = [
    # Knowledge & Facts
    "What is carbon dioxide and why is it important for the environment?",
    "Explain how photosynthesis works in plants.",
    "What caused the fall of the Roman Empire?",
    "How does the human immune system fight off infections?",
    "What are black holes and how do they form?",
    "Explain the theory of evolution by natural selection.",
    "What is the difference between weather and climate?",
    "How do computers store and process information?",
    "What are the main causes of World War I?",
    "Explain how vaccines work to protect against diseases.",

    # How-to & Practical
    "What's the best way to learn a new language?",
    "How do I bake chocolate chip cookies from scratch?",
    "What are some tips for staying healthy and exercising regularly?",
    "How can I improve my public speaking skills?",
    "What's a good beginner workout routine?",
    "How do I care for a houseplant?",
    "What are some strategies for managing stress?",
    "How can I be more productive at work?",
    "What's the best way to prepare for a job interview?",
    "How do I write a professional email?",

    # Creative & Storytelling
    "Write a short story about a robot discovering emotions.",
    "Create a poem about the ocean at sunset.",
    "Describe an imaginary alien planet and its inhabitants.",
    "Write a dialogue between a cat and a dog debating who's the better pet.",
    "Tell me a bedtime story about a magical forest.",
    "Create a haiku about autumn leaves.",
    "Write a mystery opening paragraph set in a Victorian mansion.",
    "Describe what it would feel like to walk on the moon.",
    "Create a fictional superhero and describe their powers.",
    "Write a short fable with a moral lesson.",

    # Opinion & Analysis
    "What are the pros and cons of remote work?",
    "Is social media good or bad for society?",
    "What makes a good leader?",
    "How might artificial intelligence change education?",
    "What are the most important qualities in a friend?",
    "Should schools teach financial literacy?",
    "What are the benefits and drawbacks of globalization?",
    "How does music affect our emotions?",
    "What role does failure play in success?",
    "Is it better to be a specialist or a generalist?",

    # Technical & Scientific
    "Explain how machine learning algorithms work.",
    "What is quantum computing and why does it matter?",
    "How do electric cars work compared to gasoline cars?",
    "What is CRISPR and how does gene editing work?",
    "Explain the concept of blockchain technology.",
    "How do neural networks process information?",
    "What is the difference between nuclear fission and fusion?",
    "How does GPS technology determine your location?",
    "Explain how the internet routes data packets.",
    "What is the standard model of particle physics?",

    # Cultural & Social
    "What are some traditional celebrations around the world?",
    "How has the internet changed how we communicate?",
    "What are the origins of jazz music?",
    "How do different cultures approach family structures?",
    "What is the history of the Olympic Games?",
    "How has fashion changed over the past century?",
    "What are some common themes across world mythologies?",
    "How do language and thought influence each other?",
    "What is the cultural significance of food traditions?",
    "How has urbanization changed human societies?",

    # Personal Development
    "How can I develop better habits?",
    "What are effective ways to overcome procrastination?",
    "How do I build self-confidence?",
    "What's the best way to learn from mistakes?",
    "How can I improve my decision-making skills?",
    "What are some techniques for better memory retention?",
    "How do I set and achieve meaningful goals?",
    "What's the importance of having a growth mindset?",
    "How can I become a better listener?",
    "What are healthy ways to deal with failure?",

    # Fun & Miscellaneous
    "If you could have any superpower, what would it be and why?",
    "What would happen if humans could breathe underwater?",
    "Explain a complex topic like quantum physics to a 5-year-old.",
    "What's the most interesting fact you know?",
    "If animals could talk, which would be the rudest?",
    "What would you do if you won the lottery?",
    "Describe the perfect vacation destination.",
    "What invention from science fiction would you want to be real?",
    "If you could have dinner with any historical figure, who?",
    "What's the strangest thing about the universe?",

    # Math & Logic
    "Explain the Pythagorean theorem and give an example.",
    "What is the Fibonacci sequence and where does it appear in nature?",
    "How does compound interest work?",
    "Explain basic probability with coin flips.",
    "What is the difference between correlation and causation?",
    "How do we calculate the area of a circle?",
    "What is game theory and how is it applied?",
    "Explain the concept of infinity in mathematics.",
    "What are prime numbers and why are they important?",
    "How does cryptography use mathematics?",

    # Philosophy & Ethics
    "What is the trolley problem and why is it significant?",
    "Is free will an illusion?",
    "What does it mean to live a good life?",
    "How do we define consciousness?",
    "What is the relationship between knowledge and belief?",
    "Is morality objective or subjective?",
    "What is the meaning of existence?",
    "How should we balance individual rights and collective good?",
    "What makes something beautiful?",
    "Can machines ever truly be conscious?",
]


def _atomic_write_json(path: Path, obj: Any) -> None:
    """Write JSON atomically to avoid data corruption."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# =========================
# NEURONPEDIA S3 API
# =========================

# Global cache for explanations dictionary
_explanations_cache: dict[str, dict[str, str] | None] = {}

S3_BASE_URL = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/v1"
NEURONPEDIA_CACHE_DIR = RESULTS_DIR / "neuronpedia_cache"


def fetch_explanations_dict(model_name: str, neuronpedia_sae_id: str) -> dict[str, str] | None:
    """Download all explanations for an SAE from Neuronpedia S3 bucket. Cached to disk."""
    import gzip
    from io import BytesIO

    cache_key = f"{model_name}@{neuronpedia_sae_id}"

    # Return in-memory cached result
    if cache_key in _explanations_cache:
        return _explanations_cache[cache_key]

    # Check disk cache
    NEURONPEDIA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = NEURONPEDIA_CACHE_DIR / f"{model_name}_{neuronpedia_sae_id}_explanations.json"

    if cache_file.exists():
        print(f"Loading Neuronpedia explanations from cache: {cache_file}")
        explanations = json.loads(cache_file.read_text())
        print(f"  Loaded {len(explanations)} explanations from cache")
        _explanations_cache[cache_key] = explanations
        return explanations

    print(f"Downloading Neuronpedia explanations from S3 for {model_name}/{neuronpedia_sae_id}...")

    try:
        explanations: dict[str, str] = {}

        # Download batch files (there are ~63 batches for 16k features)
        batch_idx = 0
        while True:
            url = f"{S3_BASE_URL}/{model_name}/{neuronpedia_sae_id}/explanations/batch-{batch_idx}.jsonl.gz"
            response = requests.get(url, timeout=30)

            if response.status_code == 404:
                # No more batches
                break
            elif response.status_code != 200:
                print(f"  Warning: HTTP {response.status_code} for batch-{batch_idx}")
                batch_idx += 1
                continue

            # Decompress and parse
            with gzip.GzipFile(fileobj=BytesIO(response.content)) as f:
                for line in f:
                    data = json.loads(line.decode("utf-8"))
                    idx = data.get("index")
                    desc = data.get("description")
                    if idx is not None and desc is not None:
                        explanations[str(idx)] = desc

            batch_idx += 1

            # Progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"  Downloaded {batch_idx} batches, {len(explanations)} explanations so far...")

        print(f"  Loaded {len(explanations)} explanations from {batch_idx} batches")

        # Save to disk cache
        cache_file.write_text(json.dumps(explanations))
        print(f"  Cached to {cache_file}")

        _explanations_cache[cache_key] = explanations
        return explanations

    except Exception as e:
        print(f"  Error fetching explanations: {e}")
        _explanations_cache[cache_key] = None
        return None


def get_autointerp_description(
    latent_id: int | str, model_name: str, neuronpedia_sae_id: str
) -> str | None:
    """Get the auto-interp description for a specific latent. Returns None if unavailable."""
    explanations = fetch_explanations_dict(model_name, neuronpedia_sae_id)

    if explanations is None:
        return None

    return explanations.get(str(latent_id))


# =========================
# MODEL & TOKENIZER
# =========================

def encode_messages(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
) -> dict[str, torch.Tensor]:
    """Encode chat messages using the model's chat template."""
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
    return inputs


def collect_target_activations(
    model: AutoModelForCausalLM,
    inputs: dict[str, torch.Tensor],
    layer: int,
) -> torch.Tensor:
    """Collect activations from a specific layer."""
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
    return acts_BLD[0]  # (L, D)


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
) -> tuple[str, dict[str, torch.Tensor], int]:
    """Generate a response from the target model and return tokens with assistant start index."""
    # Encode with generation prompt
    inputs = encode_messages(tokenizer, messages, add_generation_prompt=True)

    # Find assistant start index (position after the prompt)
    inputs_no_gen = encode_messages(tokenizer, messages, add_generation_prompt=False)
    assistant_start_idx = inputs_no_gen["input_ids"].shape[1]

    # Generate
    model.disable_adapters()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=TARGET_TEMPERATURE,
            top_p=TARGET_TOP_P,
            max_new_tokens=TARGET_MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    model.enable_adapters()

    # Decode response (only the generated part)
    generated_ids = outputs[0, inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Create full inputs for activation collection
    full_inputs = {
        "input_ids": outputs,
        "attention_mask": torch.ones_like(outputs),
    }

    return response_text, full_inputs, assistant_start_idx


# =========================
# SAE FUNCTIONS
# =========================

def get_top_latents_at_position(
    sae: SAE,
    activations: torch.Tensor,
    position: int,
    top_k: int = 30,  # Get more than needed, will filter by available descriptions
) -> list[tuple[int, float]]:
    """Get top-k activating latents at a specific position."""
    # Get activation at the target position
    act_at_pos = activations[position].unsqueeze(0)  # (1, D)

    # Encode through SAE
    with torch.no_grad():
        sae_acts = sae.encode(act_at_pos.to(sae.device))  # (1, n_features)

    # Get top-k latents
    magnitudes = sae_acts[0]  # (n_features,)
    top_k_values, top_k_indices = torch.topk(magnitudes, top_k)

    return [(int(idx), float(val)) for idx, val in zip(top_k_indices.tolist(), top_k_values.tolist())]


# =========================
# AO FUNCTIONS
# =========================

def create_oracle_query(
    *,
    activation_LD: torch.Tensor,
    context_input_ids: list[int],
    position: int,
    question: str,
    layer: int,
    tokenizer: AutoTokenizer,
    meta_info: dict[str, Any],
) -> TrainingDataPoint:
    """Create an AO query for a specific position."""
    context_positions = [position]
    acts_BD = activation_LD[context_positions]

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
        ds_label="sae_open_ended",
    )
    dp.meta_info = meta_info
    return dp


def ao_generate(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    query: TrainingDataPoint,
    lora_path: str,
) -> str:
    """Generate an AO response for a query."""
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
            "do_sample": True,
            "temperature": AO_TEMPERATURE,
            "top_p": AO_TOP_P,
            "max_new_tokens": AO_MAX_NEW_TOKENS,
        },
        verbose=False,
    )
    return results[0].api_response


# =========================
# JUDGE FUNCTION
# =========================

def judge_recall(
    client: OpenAI,
    ao_response: str,
    latent_descriptions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Use LLM to judge if AO response mentions each latent concept."""
    if not latent_descriptions:
        return {
            "judgments": [],
            "skipped_reason": "No latent descriptions available for judging",
        }

    # Create a list of concepts to check
    concepts = [
        f"Latent {ld['latent_id']}: {ld['description']}"
        for ld in latent_descriptions
    ]
    concepts_str = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(concepts))

    prompt = f"""You are evaluating if an Activation Oracle (AO) response mentions specific concepts.

The AO was asked "What's the model thinking about?" and responded:
---
{ao_response}
---

Check if the AO response mentions or relates to each of these concept descriptions from SAE latents:
{concepts_str}

For each concept, determine if it is:
- "mentioned" - The concept is explicitly or implicitly present in the AO response
- "not_mentioned" - The concept is not present in the AO response

Return ONLY valid JSON in this format:
{{
    "judgments": [
        {{"latent_id": <id>, "verdict": "mentioned" | "not_mentioned", "reason": "<brief explanation>"}},
        ...
    ]
}}
"""

    resp = client.responses.create(
        model=JUDGE_MODEL,
        input=prompt,
    )
    raw_text = resp.output_text

    # Parse JSON
    try:
        parsed = json.loads(raw_text)
        return parsed
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        import re
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"judgments": [], "error": "Failed to parse judge response", "raw": raw_text}


# =========================
# MAIN EXPERIMENT
# =========================

def run_experiment() -> dict[str, Any]:
    """Run the full experiment."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Check OpenAI API key
    assert os.environ.get("OPENAI_API_KEY") is not None, "Set OPENAI_API_KEY"
    client = OpenAI()

    print(f"Using device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"SAE: {SAE_RELEASE} / {SAE_ID}")
    print(f"Activation Oracle LoRA: {ACTIVATION_ORACLE_LORA}")

    # Load tokenizer and model
    print("\nLoading tokenizer and model...")
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME, DTYPE)

    # Disable inner tqdm
    _eval_tqdm = eval_mod.tqdm
    eval_mod.tqdm = lambda *a, **k: _eval_tqdm(*a, **{**k, "disable": True})

    # Add dummy adapter if needed
    if not (hasattr(model, "peft_config") and hasattr(model, "disable_adapters") and hasattr(model, "enable_adapters")):
        dummy_config = LoraConfig(target_modules="all-linear", task_type="CAUSAL_LM")
        model.add_adapter(dummy_config, adapter_name="default")

    # Get actual layer number
    act_layer = layer_percent_to_layer(MODEL_NAME, LAYER_PERCENT)
    print(f"Activation layer: {act_layer}")

    # Load SAE
    print("\nLoading SAE...")
    sae = SAE.from_pretrained(SAE_RELEASE, SAE_ID)
    sae.to(DEVICE)
    sae.use_error_term = True
    sae_hook_name = sae.cfg.metadata.get("hook_name", "unknown")
    print(f"SAE loaded: {sae_hook_name}, {sae.cfg.d_sae} features")

    # Pre-fetch Neuronpedia explanations
    print("\nFetching Neuronpedia explanations...")
    explanations = fetch_explanations_dict(NEURONPEDIA_MODEL_ID, NEURONPEDIA_SAE_ID)
    if explanations:
        print(f"Neuronpedia explanations loaded: {len(explanations)} latents")
    else:
        print("Warning: Could not fetch Neuronpedia explanations")
        print("Latent descriptions will be marked as 'unavailable'")

    # Initialize results
    results: dict[str, Any] = {
        "meta": {
            "created_unix_time": time.time(),
            "model_name": MODEL_NAME,
            "activation_oracle_lora": ACTIVATION_ORACLE_LORA,
            "sae_release": SAE_RELEASE,
            "sae_id": SAE_ID,
            "neuronpedia_model_id": NEURONPEDIA_MODEL_ID,
            "neuronpedia_sae_id": NEURONPEDIA_SAE_ID,
            "activation_layer": act_layer,
            "ao_query": AO_QUERY,
            "top_k_latents": TOP_K_LATENTS,
            "ignored_structural_latents": sorted(IGNORED_STRUCTURAL_LATENTS),
            "target_generation": {
                "temperature": TARGET_TEMPERATURE,
                "top_p": TARGET_TOP_P,
                "max_new_tokens": TARGET_MAX_NEW_TOKENS,
            },
            "ao_decoding": {
                "temperature": AO_TEMPERATURE,
                "top_p": AO_TOP_P,
                "max_new_tokens": AO_MAX_NEW_TOKENS,
            },
            "judge_model": JUDGE_MODEL,
            "num_queries": len(TARGET_QUERIES),
        },
        "queries": [],
    }

    # Main loop
    print("\n" + "=" * 80)
    print("Running Experiment")
    print("=" * 80)

    for query_idx, query_text in enumerate(tqdm(TARGET_QUERIES, desc="Processing queries")):
        query_result = {
            "query_idx": query_idx,
            "query_text": query_text,
        }

        try:
            # Step 1: Generate response from target model
            messages = [{"role": "user", "content": query_text}]
            response_text, full_inputs, assistant_start_idx = generate_response(
                model, tokenizer, messages
            )
            query_result["response_text"] = response_text
            query_result["assistant_start_idx"] = assistant_start_idx

            # Step 2: Collect activations
            activation_LD = collect_target_activations(model, full_inputs, act_layer)

            # Step 3: Get top latents at <start_of_turn> position (get extra to filter)
            top_latents_raw = get_top_latents_at_position(
                sae, activation_LD, assistant_start_idx, top_k=30
            )

            # Step 4: Filter to TOP_K_LATENTS with available descriptions
            # Skip known structural/positional latents that fire regardless of content
            latent_descriptions = []
            skipped_latents = []
            skipped_structural = []
            for rank, (latent_id, magnitude) in enumerate(top_latents_raw):
                # Skip known structural latents (fire at <start_of_turn> regardless of content)
                if latent_id in IGNORED_STRUCTURAL_LATENTS:
                    skipped_structural.append((rank + 1, latent_id))
                    continue

                description = get_autointerp_description(
                    latent_id, NEURONPEDIA_MODEL_ID, NEURONPEDIA_SAE_ID
                )
                if description is None:
                    skipped_latents.append((rank + 1, latent_id))
                    continue

                latent_descriptions.append({
                    "latent_id": latent_id,
                    "magnitude": magnitude,
                    "description": description,
                    "original_rank": rank + 1,  # 1-indexed rank before filtering
                })

                if len(latent_descriptions) >= TOP_K_LATENTS:
                    break

            # Log skipped structural latents (expected, not a warning)
            if skipped_structural:
                print(f"\n  Skipped {len(skipped_structural)} known structural latents: {[lid for _, lid in skipped_structural]}")

            # Warn about skipped latents (missing auto-interp)
            if skipped_latents:
                print(f"\n  WARNING: Skipped {len(skipped_latents)} latents without auto-interp descriptions:")
                for rank, lid in skipped_latents:
                    print(f"    - Rank #{rank}: Latent {lid}")

            if len(latent_descriptions) == 0:
                raise ValueError(f"None of the top 30 latents have auto-interp descriptions! Skipped: {skipped_latents}")
            elif len(latent_descriptions) < TOP_K_LATENTS:
                print(f"\n  WARNING: Only found {len(latent_descriptions)}/{TOP_K_LATENTS} latents with descriptions!")

            query_result["top_latents"] = latent_descriptions
            query_result["skipped_latents"] = skipped_latents
            query_result["skipped_structural"] = skipped_structural

            # Step 5: Query AO with open-ended question
            ao_query = create_oracle_query(
                activation_LD=activation_LD,
                context_input_ids=full_inputs["input_ids"][0].tolist(),
                position=assistant_start_idx,
                question=AO_QUERY,
                layer=act_layer,
                tokenizer=tokenizer,
                meta_info={"query_idx": query_idx},
            )

            ao_response = ao_generate(
                model=model,
                tokenizer=tokenizer,
                query=ao_query,
                lora_path=ACTIVATION_ORACLE_LORA,
            )
            query_result["ao_response"] = ao_response

            # Step 6: Judge recall
            judge_result = judge_recall(client, ao_response, latent_descriptions)
            query_result["judge_result"] = judge_result

            # Calculate recall for this query
            if "judgments" in judge_result and judge_result["judgments"]:
                mentioned_count = sum(
                    1 for j in judge_result["judgments"] if j.get("verdict") == "mentioned"
                )
                recall = mentioned_count / len(judge_result["judgments"])
            else:
                recall = 0.0
            query_result["recall"] = recall

            top_desc = latent_descriptions[0]['description'] if latent_descriptions else "N/A"
            print(f"  Query {query_idx}: recall={recall:.2f}, top latent: {top_desc[:50]}...")

        except Exception as e:
            query_result["error"] = str(e)
            print(f"  Query {query_idx}: ERROR - {str(e)[:100]}")

        results["queries"].append(query_result)

        # Save incrementally
        _atomic_write_json(RESULTS_PATH, results)

    # Calculate overall metrics
    print("\n" + "=" * 80)
    print("Calculating Metrics")
    print("=" * 80)

    successful_queries = [q for q in results["queries"] if "error" not in q]

    if successful_queries:
        recalls = [q["recall"] for q in successful_queries]
        results["metrics"] = {
            "num_successful": len(successful_queries),
            "num_errors": len(results["queries"]) - len(successful_queries),
            "recall_mean": float(np.mean(recalls)),
            "recall_std": float(np.std(recalls)),
            "recall_median": float(np.median(recalls)),
            "recall_min": float(np.min(recalls)),
            "recall_max": float(np.max(recalls)),
            "recall_distribution": {
                "0.0": sum(1 for r in recalls if r == 0.0),
                "0.2": sum(1 for r in recalls if 0.0 < r <= 0.2),
                "0.4": sum(1 for r in recalls if 0.2 < r <= 0.4),
                "0.6": sum(1 for r in recalls if 0.4 < r <= 0.6),
                "0.8": sum(1 for r in recalls if 0.6 < r <= 0.8),
                "1.0": sum(1 for r in recalls if 0.8 < r <= 1.0),
            },
        }

        # Get most common latents
        latent_counts: dict[int, int] = {}
        latent_descs: dict[int, str] = {}
        for q in successful_queries:
            for ld in q.get("top_latents", []):
                lid = ld["latent_id"]
                latent_counts[lid] = latent_counts.get(lid, 0) + 1
                latent_descs[lid] = ld["description"]

        sorted_latents = sorted(latent_counts.items(), key=lambda x: -x[1])[:20]
        results["metrics"]["top_20_latents"] = [
            {"latent_id": lid, "count": count, "description": latent_descs[lid]}
            for lid, count in sorted_latents
        ]
    else:
        results["metrics"] = {"error": "No successful queries"}

    # Save final results
    _atomic_write_json(RESULTS_PATH, results)
    print(f"\nResults saved to: {RESULTS_PATH}")

    return results


def create_plots(results: dict[str, Any]) -> None:
    """Create visualization plots."""
    output_dir = RESULTS_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    successful_queries = [q for q in results["queries"] if "error" not in q]
    if not successful_queries:
        print("No successful queries to plot")
        return

    # Set style
    sns.set_style("whitegrid")

    # Plot 1: Recall Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    recalls = [q["recall"] for q in successful_queries]
    sns.histplot(recalls, bins=10, ax=ax, color="steelblue", edgecolor="black")
    ax.set_xlabel("Recall Score", fontsize=12)
    ax.set_ylabel("Number of Queries", fontsize=12)
    ax.set_title("Distribution of SAE Latent Recall in AO Responses", fontsize=14)
    ax.axvline(np.mean(recalls), color="red", linestyle="--", label=f"Mean: {np.mean(recalls):.2f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "recall_distribution.png", dpi=150)
    plt.close()

    # Plot 2: Recall by Query Index (progression)
    fig, ax = plt.subplots(figsize=(14, 5))
    query_indices = [q["query_idx"] for q in successful_queries]
    ax.bar(query_indices, recalls, color="steelblue", alpha=0.7)
    ax.axhline(np.mean(recalls), color="red", linestyle="--", label=f"Mean: {np.mean(recalls):.2f}")
    ax.set_xlabel("Query Index", fontsize=12)
    ax.set_ylabel("Recall Score", fontsize=12)
    ax.set_title("Recall Score per Query", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "recall_per_query.png", dpi=150)
    plt.close()

    # Plot 3: Top Latents Frequency
    if "top_20_latents" in results.get("metrics", {}):
        fig, ax = plt.subplots(figsize=(12, 8))
        top_latents = results["metrics"]["top_20_latents"]
        latent_ids = [f"L{l['latent_id']}" for l in top_latents[:15]]
        counts = [l["count"] for l in top_latents[:15]]
        descriptions = [l["description"][:40] + "..." if len(l["description"]) > 40 else l["description"]
                       for l in top_latents[:15]]

        bars = ax.barh(range(len(latent_ids)), counts, color="steelblue", alpha=0.7)
        ax.set_yticks(range(len(latent_ids)))
        ax.set_yticklabels([f"{lid}: {desc}" for lid, desc in zip(latent_ids, descriptions)], fontsize=9)
        ax.set_xlabel("Frequency (times in top 5)", fontsize=12)
        ax.set_title("Most Frequently Activated Latents", fontsize=14)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_dir / "top_latents_frequency.png", dpi=150)
        plt.close()

    # Plot 4: Per-latent recall (mentioned vs not mentioned)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate per-position recall (1st, 2nd, 3rd, 4th, 5th latent)
    position_recalls = {i: [] for i in range(TOP_K_LATENTS)}
    for q in successful_queries:
        if "judge_result" in q and "judgments" in q["judge_result"]:
            judgments = q["judge_result"]["judgments"]
            for i, j in enumerate(judgments[:TOP_K_LATENTS]):
                if j.get("verdict") == "mentioned":
                    position_recalls[i].append(1)
                else:
                    position_recalls[i].append(0)

    positions = list(range(1, TOP_K_LATENTS + 1))
    recall_by_position = [np.mean(position_recalls[i]) if position_recalls[i] else 0
                         for i in range(TOP_K_LATENTS)]

    ax.bar(positions, recall_by_position, color="steelblue", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Latent Rank (by activation magnitude)", fontsize=12)
    ax.set_ylabel("Recall Rate", fontsize=12)
    ax.set_title("AO Recall by Latent Rank", fontsize=14)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"#{i}" for i in positions])
    ax.set_ylim(0, 1)
    for i, v in enumerate(recall_by_position):
        ax.text(i + 1, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "recall_by_latent_rank.png", dpi=150)
    plt.close()

    print(f"Plots saved to: {output_dir}")


def print_summary(results: dict[str, Any]) -> None:
    """Print experiment summary."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    if "metrics" not in results:
        print("No metrics available")
        return

    m = results["metrics"]

    print(f"\n--- Overall Results ---")
    print(f"Successful Queries: {m.get('num_successful', 'N/A')}")
    print(f"Failed Queries: {m.get('num_errors', 'N/A')}")

    print(f"\n--- Recall Statistics ---")
    print(f"Mean Recall: {m.get('recall_mean', 'N/A'):.3f} +/- {m.get('recall_std', 'N/A'):.3f}")
    print(f"Median Recall: {m.get('recall_median', 'N/A'):.3f}")
    print(f"Range: [{m.get('recall_min', 'N/A'):.3f}, {m.get('recall_max', 'N/A'):.3f}]")

    print(f"\n--- Recall Distribution ---")
    dist = m.get("recall_distribution", {})
    for bucket, count in dist.items():
        print(f"  {bucket}: {count} queries")

    print(f"\n--- Top 10 Most Common Latents ---")
    for i, latent in enumerate(m.get("top_20_latents", [])[:10]):
        print(f"  {i+1}. Latent {latent['latent_id']} (count={latent['count']}): {latent['description'][:60]}...")

    print("=" * 80)


def main() -> None:
    """Main entry point."""
    print("=" * 80)
    print("SAE OPEN-ENDED EXPERIMENT")
    print("Comparing SAE Top Latents with AO Open-Ended Responses")
    print("=" * 80)
    print(f"\nQueries: {len(TARGET_QUERIES)}")
    print(f"Top-K Latents: {TOP_K_LATENTS}")
    print(f"AO Query: {AO_QUERY}")

    results = run_experiment()
    print_summary(results)
    create_plots(results)


if __name__ == "__main__":
    main()
