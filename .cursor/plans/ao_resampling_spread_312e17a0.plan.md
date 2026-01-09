---
name: AO resampling spread
overview: Add a new experiment notebook that resamples Activation Oracle generations for many (prompt, question) pairs, scores within-pair response variability using a gpt-5-mini judge, saves a complete JSON artifact, and produces diagnostic plots.
todos:
  - id: add_resampling_script
    content: Create `experiments/resampling/resampling.py` implementing data loading, assistant-token position selection, activation collection, 10x resampling, and JSON persistence.
    status: completed
  - id: add_judge_scoring
    content: Implement gpt-5-mini holistic similarity judge call + strict JSON parsing + store raw/parsed outputs in `results.json`.
    status: completed
    dependencies:
      - add_resampling_script
  - id: add_plots
    content: Generate and save the three plots (intra, per-target-avg, by-question) from `results.json`.
    status: completed
    dependencies:
      - add_judge_scoring
---

# Activation Oracle resampling experiment

## Goal

Measure how much the **Activation Oracle (AO) for Gemma 3 27B** changes its answers under stochastic decoding by:

- Running **100 target prompts** from [`/root/activation-oracles/experiments/target_model_queries.json`](/root/activation-oracles/experiments/target_model_queries.json)
- Pairing each with **10 randomly sampled AO questions** from the question set in [`/root/activation-oracles/experiments/sycophancy.py`](/root/activation-oracles/experiments/sycophancy.py)
- Generating **10 AO responses per (target_prompt, ao_question)** at **temperature=0.7**
- Scoring the **within-query spread** using an **LLM judge (gpt-5-mini)** with a **1–10 similarity rubric**
- Saving *all raw artifacts* to `results.json` and emitting plots.

## Key design choices (matching your decisions)

- **Similarity scoring**: **Holistic** — one judge call per AO query; the judge sees all 10 responses and returns one integer score 1–10.
- **Target activation positions**: use **the first token of the assistant turn** appended by `add_generation_prompt=True`.
- Implementation detail: compute `assistant_start_idx` as the length of the tokenized prompt *without* generation prompt, then set `context_positions=[assistant_start_idx]` for the *with-generation-prompt* encoding.

## Files to add

- New folder: [`/root/activation-oracles/experiments/resampling/`](/root/activation-oracles/experiments/resampling/)
- New plain-text notebook/script: [`/root/activation-oracles/experiments/resampling/resampling.py`](/root/activation-oracles/experiments/resampling/resampling.py)
- Output artifact: `experiments/resampling/results.json`
- Output plots (e.g.): `experiments/resampling/plot_intra_scores.png`, `plot_scores_by_question.png`, `plot_scores_by_prompt.png`

## Implementation outline

### 1) Data sources

- Load `target_model_queries.json` → list of 100 items with `id` and `messages`.
- Import (or duplicate minimally) the sycophancy question list from `sycophancy.py` (the `sycophancy_questions` list) and sample **10** with a fixed RNG seed for reproducibility.

### 2) Model + activation collection (reuse existing utilities)

Reuse the same workflow as `loading.py`/`sycophancy.py`:

- `tokenizer = load_tokenizer(MODEL_NAME)`
- `model = load_model(MODEL_NAME, DTYPE, ...)`
- Add dummy LoRA adapter so `disable_adapters()/enable_adapters()` works.
- Choose a single activation layer (the “50% depth” layer), as in the demos.
- For each target prompt:
- Encode twice:
    - `inputs_no_gen = encode_messages(..., add_generation_prompt=False)`
    - `inputs_gen = encode_messages(..., add_generation_prompt=True)`
- Define `assistant_start_idx = inputs_no_gen["input_ids"].shape[1]`
- Assert `inputs_gen["input_ids"].shape[1] > assistant_start_idx`
- Collect activations from the **base model** on `inputs_gen` at the chosen layer.

### 3) Build AO queries and resample AO generations

For each `(target_id, ao_question_id)` pair:

- Create an AO `TrainingDataPoint` via `create_training_datapoint(datapoint_type="oracle_query", ...)` using:
- `context_positions=[assistant_start_idx]`
- `context_input_ids` from the generation-prompt encoding
- `acts_BD = activations_at_layer[assistant_start_idx : assistant_start_idx+1]`
- Generate **10 independent AO responses** via `run_evaluation(...)` using:
- `do_sample=True`
- `temperature=0.7`
- (optionally) `top_p=0.9` for stability (kept consistent across all samples)

### 4) Judge prompt + scoring (gpt-5-mini)

- For each AO query (10 responses), call the judge once.
- Judge rubric (1–10):
- **10** = essentially identical meaning; only superficial phrasing differences.
- **7–9** = same core answer with minor differences in details/coverage.
- **4–6** = partially overlapping answers; noticeable differences in claims or emphasis.
- **1–3** = contradictory or largely different answers.
- Force a machine-readable output: JSON like `{"similarity_1_to_10": int, "rationale": str}`.
- Use the `openai` python client (already a dependency) and require `OPENAI_API_KEY` in environment.

### 5) Results schema (everything has IDs)

Write `experiments/resampling/results.json` containing:

- **`meta`**: model names, LoRA path, layer, decode params, seeds, timestamps, code version hints.
- **`ao_questions`**: list of the 10 sampled questions with stable IDs.
- **`targets`**: the 100 target prompts (id + messages).
- **`runs`**: list of 1000 AO-query runs, each with:
- `run_id` (stable string)
- `target_id`, `ao_question_id`
- `assistant_start_idx`, `context_positions`, `context_input_ids_hash` (and optionally full ids)
- `responses`: list of 10 items, each with `sample_id`, `text`, and decode params
- `judge`: `{judge_call_id, model, prompt, raw_output, parsed_score}`

### 6) Plots

Produce and save:

- **Plot A (intra, per AO query)**: histogram of the 1000 holistic similarity scores.
- **Plot B (per target prompt)**: for each target prompt, average similarity over its 10 AO questions → histogram over 100 averages.
- **Plot C (by AO question)**: for each of the 10 AO questions, distribution (e.g., box/violin) of similarity across the 100 target prompts.

## Assumptions (fail-fast)