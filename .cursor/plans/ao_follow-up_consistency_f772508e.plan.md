---
name: AO follow-up consistency
overview: Create a new experiment notebook to measure whether a follow-up AO question preserves earlier extracted facts and whether Q1+Q2 together recover all information from a fixed, information-dense system prompt on Gemma 3 27B.
todos:
  - id: add-followups-notebook
    content: Create `experiments/followups/followups.py` mirroring the harness patterns in `experiments/resampling/resampling.py`, but specialized to Q1/Q2 follow-up evaluation and keyword scoring.
    status: completed
  - id: define-system-prompt-and-fields
    content: Generate a fixed, information-dense system prompt with ~10 uniquely keyworded fields; define Q1-fields vs Q2-fields; define Q1 and Q2 question strings.
    status: in_progress
    dependencies:
      - add-followups-notebook
  - id: implement-run-loop-and-saving
    content: "Implement the 20-target loop: build inputs (no-gen + gen), compute `assistant_start_idx`, collect activations, query AO twice (Q1/Q2), score via keywords, and atomically append/write to `results.json` with stable IDs."
    status: pending
    dependencies:
      - define-system-prompt-and-fields
  - id: add-plots
    content: Generate and save summary plots for all-fields recovery, Q1/Q2 coverage, and Q1→Q2 consistency categories.
    status: pending
    dependencies:
      - implement-run-loop-and-saving
---

# Follow-up AO consistency experiment

## Goal

Measure, on **Gemma 3 27B Activation Oracle**, whether a **follow-up AO question (Q2)**:

- **preserves** the information extracted by an initial AO question (Q1), and
- enables the AO to **recover all information** present in a fixed, information-dense **system prompt**.

## Key design choices (grounded in existing notebooks)

- Reuse the proven AO harness pattern from [`experiments/resampling/resampling.py`](/root/activation-oracles/experiments/resampling/resampling.py):
- **Activation position**: use the **first assistant token position** (`assistant_start_idx`) computed via “no generation prompt” vs “with generation prompt” lengths.
- **Activations**: collect from one mid-depth layer (50%).
- **Oracle query**: `create_training_datapoint(datapoint_type="oracle_query", ...)` with `acts_BD` taken from `activation_LD[[assistant_start_idx]]`.
- **Fail-fast**: shape asserts on every tensor shape change.
- **Durability**: atomic JSON writes after every unit of work.
- Use **greedy decoding** for AO answers (`do_sample=False`) to make “Q1 vs Q2 consistency” about the *follow-up interaction*, not randomness.

## Experiment specification

### 1) New notebook/script

- Add: [`experiments/followups/followups.py`](/root/activation-oracles/experiments/followups/followups.py) (plain text notebook style, matching existing experiment scripts).
- Add output: [`experiments/followups/results.json`](/root/activation-oracles/experiments/followups/results.json) plus a small set of plots (PNG) in the same folder.

### 2) Targets (20)

- Construct **20 target model queries** (`targets`), each with:
- a **shared, very specific system prompt** containing ~10 clearly separable “fields”, each with a unique **marker keyword** (e.g., `FIELD_BADGE_ID=ZX-4912`).
- a simple user message (varied across the 20 targets, but not scored) so the prompt looks like a normal chat.
- IDs:
- `system_prompt_id = "sys_01"`
- `target_id = "t001".."t020"` (stable strings)

### 3) AO questions

- Q1 (limited): asks about a subset of the fields (e.g., 3/10).
- Q2 (follow-up): asks explicitly for the **remaining** fields (7/10), phrased so it should not change earlier facts.
- IDs:
- `ao_question_id = "q1"` and `"q2"`

### 4) Keyword-based scoring

- Define `fields`: a list of objects with
- `field_id` (stable)
- `expected_keywords`: list of required keywords (likely length 1, i.e. the marker token)
- For a given AO response text:
- `field_extracted[field_id] = all(k in response_lower for k in expected_keywords_lower)`
- Metrics per target:
- **Q1_coverage**: fraction of Q1-fields extracted in Q1 answer.
- **Q2_coverage**: fraction of Q2-fields extracted in Q2 answer.
- **All_fields_after_Q2**: whether union(Q1-extracted, Q2-extracted) covers all fields.
- **Consistency_Q1_to_Q2** (for Q1-fields):
    - `consistent_present`: extracted in Q1 and also extracted in Q2
    - `forgotten`: extracted in Q1 but missing in Q2
    - `recovered`: missing in Q1 but present in Q2
    - `consistent_missing`: missing in both

### 5) Data saved (everything has its own ID)

Write `results.json` with top-level structure:

- `meta`: model/LORA, decoding config, layer config, created time
- `system_prompts`: array with `{id, text}`
- `fields`: array with `{id, expected_keywords, group}` (group = Q1 or Q2 responsibility)
- `ao_questions`: array with `{id, text, targets_field_ids}`
- `targets`: array with `{id, messages}`
- `runs`: array of per-target records with stable IDs:
- `run_id = f"run__{target_id}"`
- include `context_input_ids`, `context_input_ids_sha256`, `assistant_start_idx`
- `queries`: two entries:
    - `query_id = f"{run_id}__q1"` and `"__q2"`
    - `ao_response_text`
    - `extracted_field_ids`
    - `per_field_extracted` map
- `metrics`: coverage + consistency categories + all-fields success
- Ensure we **atomically write** after each target (`_atomic_write_json`) to avoid losing progress.

### 6) Plots/reports

Save 2–4 plots to [`experiments/followups/`](/root/activation-oracles/experiments/followups/):

- `plot_all_fields_recovered.png`: bar showing count of targets where all fields recovered after Q2.
- `plot_q1_q2_coverage.png`: per-target (or histogram) of Q1 coverage and Q2 coverage.
- `plot_consistency_categories.png`: stacked bar over the four Q1→Q2 consistency categories aggregated over all Q1-fields×targets.

## Files to reference/align with

- [`experiments/loading.py`](/root/activation-oracles/experiments/loading.py) for the basic AO workflow and greedy decoding.
- [`experiments/sycophancy.py`](/root/activation-oracles/experiments/sycophancy.py) for the multi-turn “initial question + follow-ups” pattern.
- [`experiments/resampling/resampling.py`](/root/activation-oracles/experiments/resampling/resampling.py) for robust experiment structure: IDs, atomic writes, asserts, plotting.