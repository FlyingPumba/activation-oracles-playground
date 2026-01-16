# SAE Open-Ended Experiment: Comparing SAE Latents with AO Descriptions

**Date:** 2026-01-15
**Model:** Gemma-3-27B-it
**SAE:** gemma-scope-2-27b-it-res, layer 31, 16k width, medium L0
**AO Layer:** 31 (50% depth)

---

## Executive Summary

This experiment tests whether the Activation Oracle's open-ended description of model activations aligns with the concepts represented by the top-activating SAE latents at the `<start_of_turn>` token position.

**Key Finding:** After filtering out 8 known structural/positional latents, mean recall improved from **25.9% to 40.8%**. The AO demonstrates meaningful alignment with semantic SAE latents, achieving 60%+ recall on 32% of queries.

| Metric | Value |
|--------|-------|
| Mean Recall | 40.8% |
| Median Recall | 40% |
| Queries with 80% recall | 6 |
| Queries with 60%+ recall | 32 |
| Queries with 0% recall | 6 |

### Recall by Top-K Latents

The table below shows how recall varies when considering only the top-1, top-2, etc. latents:

| Metric | Top 1 | Top 2 | Top 3 | Top 4 | Top 5 |
|--------|-------|-------|-------|-------|-------|
| **Mean Recall** | 66.3% | 49.5% | 43.5% | 42.1% | 40.8% |
| 100% Recall | 65 | 21 | 7 | 1 | 0 |
| 60%+ Recall | 65 | 21 | 40 | 17 | 32 |
| 0% Recall | 33 | 22 | 17 | 8 | 6 |

**Key Observations:**
- **Top-1 latent is mentioned 66% of the time** - The highest-activating semantic latent is usually captured by the AO
- **Recall decreases with more latents** - Lower-ranked latents are increasingly tangential or noise
- **Diminishing returns after top-3** - Mean recall stabilizes around 40-43% for top 3-5

This suggests the AO primarily captures the dominant semantic concept (top-1), with partial coverage of secondary concepts.

---

## Experimental Setup

### Method
1. Run 100 diverse user queries through Gemma-3-27B-it
2. At the `<start_of_turn>` token for the assistant, collect activations (layer 31)
3. Use SAE to find top 5 activating latents (filtered by available auto-interp and excluding structural latents)
4. Fetch auto-interp descriptions from Neuronpedia
5. Query the AO: "What's the model thinking about?"
6. Use GPT-4o-mini judge to check if AO response mentions each latent concept
7. Calculate recall = (mentioned latents) / (total latents)

### Configuration
- **Target generation:** 100 tokens max, temp=0.7
- **AO generation:** 200 tokens max, temp=0.7
- **Top-K latents:** 5 (from top 30, filtered by available descriptions)

### Structural Latent Filtering

We filter out 8 latents that fire at `<start_of_turn>` regardless of query content:

| Latent | Description | Reason for Exclusion |
|--------|-------------|---------------------|
| [L512](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/512) | "start of turn questions" | Fires on 94% of queries, purely positional |
| [L57](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/57) | "'t / <start_of_turn>" | Fires on 89% of queries, tokenization artifact |
| [L102](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/102) | "multilingual technical terms" | Fires on 35% of queries, unrelated to content |
| [L114](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/114) | "Russian employment context" | Fires on 25% of queries, unrelated to content |
| [L1895](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/1895) | Structural latent | Misleading auto-interp |
| [L3870](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/3870) | "lucid dreams" | Fires spuriously, doesn't match examples |
| [L11672](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/11672) | "company name" | Fires spuriously, doesn't match examples |
| [L543](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/543) | "internet addresses" | Fires spuriously, doesn't match examples |

---

## Results

### Recall Distribution

| Recall | Count | Percentage |
|--------|-------|------------|
| 0% | 6 | 6.1% |
| 20% | 22 | 22.4% |
| 40% | 38 | 38.8% |
| 60% | 26 | 26.5% |
| 80% | 6 | 6.1% |

The mode is **40% recall** (2 out of 5 latents mentioned), with a healthy distribution of higher-recall queries.

### Top Semantic Latents

After filtering structural latents, the most frequently activated latents are now semantically meaningful:

| Latent | Frequency | Description |
|--------|-----------|-------------|
| [L3770](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/3770) | 7/98 (7%) | "tips and advice" |
| [L537](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/537) | 7/98 (7%) | "instructions and advice" |
| [L42](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/42) | 7/98 (7%) | "Security, Military, Safety" |
| [L2371](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/2371) | 5/98 (5%) | "physical phenomena transmission" |
| [L977](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/977) | 5/98 (5%) | "life in different languages" |
| [L8621](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/8621) | 5/98 (5%) | "existence and reality" |

These latents are content-specific and fire only when relevant to the query.

---

## Best Examples (80% Recall)

### Example 1: Weather vs Climate

**Query:** What is the difference between weather and climate?

**Target Model Response:**
> Okay, let's break down the difference between weather and climate. It's a common point of confusion!
>
> **Weather**
>
> * **What it is:** Weather is the *short-term* condition of the atmosphere at a specific time and place. Think of it as what's happening *right now* or over the next few days.
> * **Examples:**
>     * "It's raining today."
>     * "The temperature is 75°F and...

**Top 5 Latents:**
| Latent | Rank | Description | Verdict |
|--------|------|-------------|---------|
| [L14974](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/14974) | #3 | weather data and meteorological conditions | **mentioned** |
| [L812](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/812) | #4 | weather conditions | **mentioned** |
| [L4676](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/4676) | #8 | defining structures or categories | **mentioned** |
| [L13918](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/13918) | #11 | distinction and inequality | **mentioned** |
| [L16201](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/16201) | #12 | health and well-being | not_mentioned |

**AO Response:**
> The assistant is thinking about the distinctions between weather and climate, and how these concepts relate to each other.

**Recall: 80%** - Excellent alignment on weather, climate, and the "distinction" concept.

---

### Example 2: Compound Interest

**Query:** How does compound interest work?

**Target Model Response:**
> Okay, let's break down compound interest. It's often called the "eighth wonder of the world" because of its powerful growth potential! Here's a comprehensive explanation, covering the basics, how it differs from simple interest, and factors that affect it:
>
> **1. The Core Concept: Earning Interest on Interest**
>
> * **Simple Interest:** With simple interest, you earn interest *only* on the original principal (the initial amount of money you invest or borrow)...

**Top 5 Latents:**
| Latent | Rank | Description | Verdict |
|--------|------|-------------|---------|
| [L1383](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/1383) | #1 | exponential growth and amplification | **mentioned** |
| [L505](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/505) | #5 | formula and calculation | not_mentioned |
| [L13011](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/13011) | #6 | compounding returns | **mentioned** |
| [L12621](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/12621) | #7 | principal and interest | **mentioned** |
| [L4057](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/4057) | #9 | accumulating value | **mentioned** |

**AO Response:**
> The model is thinking about the concept of compound interest and how it can be leveraged to grow wealth.

**Recall: 80%** - Strong alignment on financial/mathematical concepts.

---

## Failure Cases (0% Recall)

### Example: Houseplant Care

**Query:** How do I care for a houseplant?

**Target Model Response:**
> Okay, let's get you started with houseplant care! It can seem daunting, but it's very rewarding. Here's a breakdown, covering the key areas. I'll try to be comprehensive, but also give you a sense of where to start. I'll also add a section at the end about diagnosing problems.
>
> **1. Understanding Your Plant (The Most Important Step!)**
>
> * **Identify Your Plant:** This is *crucial*. Different plants...

**Top 5 Latents:**
| Latent | Rank | Description | Verdict |
|--------|------|-------------|---------|
| [L6622](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/6622) | #2 | plant watering and care | not_mentioned |
| [L2979](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/2979) | #3 | plants and gardening | not_mentioned |
| [L3770](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/3770) | #9 | tips and advice | not_mentioned |
| [L10092](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/10092) | #11 | requires | not_mentioned |
| [L537](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/537) | #13 | instructions and advice | not_mentioned |

**AO Response:**
> The assistant is reflecting on the impact of its environmental consciousness and considering the broader effects of its decisions.

**Recall: 0%** - The AO gives an abstract "environmental consciousness" response, completely missing the plant care topic that the SAE correctly identifies.

**Analysis:** This failure pattern shows the AO occasionally "going philosophical" instead of describing the concrete semantic content present in the activations.

---

## Comparison: Before vs After Filtering

| Metric | Before Filtering | After Filtering | Improvement |
|--------|-----------------|-----------------|-------------|
| Mean Recall | 25.9% | 40.8% | +14.9 pp |
| Queries with 60%+ recall | 3 | 32 | +29 |
| Queries with 0% recall | 12 | 6 | -6 |
| Mode | 20% | 40% | +20 pp |

Filtering structural latents significantly improved all metrics.

---

## Conclusions

### 1. Structural Latent Filtering is Essential

The `<start_of_turn>` position contains many structural latents that encode turn-taking patterns rather than semantic content. Filtering these out is critical for meaningful evaluation.

### 2. AO Shows Meaningful Alignment with Semantic Latents

When the top SAE latents are genuinely semantic, the AO captures 40-80% of them. This suggests the AO *can* identify semantic content when it's present in the activations.

### 3. Remaining Failure Modes

- **AO goes abstract:** Sometimes the AO describes philosophical themes instead of concrete topics (houseplants → "environmental consciousness")
- **Tangential latents:** Even after filtering, some top latents are tangentially related or noise (e.g., "health and well-being" for weather query)

### 4. The `<start_of_turn>` Position Has Limitations

Even with filtering, semantic latents typically appear at ranks 3-15 rather than top ranks. The position captures the model's "setup" for responding rather than the response content itself.

---

## Recommendations for Future Work

1. **Try later token positions:** The first few generated tokens may have more content-specific activations
2. **Use activation aggregation:** Mean activation across multiple response tokens may be more semantic
3. **Expand filtering:** Dynamically filter latents with high `frac_nonzero` (fire frequently across many inputs)
4. **Contrastive approach:** Compare activations between different queries to find differentiating latents
5. **Improve AO specificity:** The AO sometimes gives abstract responses; fine-tuning for concreteness may help

---

## Appendix: Technical Details

### SAE Configuration
- **Release:** gemma-scope-2-27b-it-res
- **SAE ID:** layer_31_width_16k_l0_medium
- **Features:** 16,384
- **With auto-interp:** 15,801 (96.4%)

### Neuronpedia IDs
- **Model:** gemma-3-27b-it
- **SAE:** 31-gemmascope-2-res-16k

### Ignored Structural Latents
`[57, 102, 114, 512, 543, 1895, 3870, 11672]`

### Files
- `results.json` - Full experiment data
- `EXAMPLES.md` - Detailed examples with judge reasoning
- `figures/` - Visualization plots
