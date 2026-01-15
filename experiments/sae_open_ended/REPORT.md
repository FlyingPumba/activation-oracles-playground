# SAE Open-Ended Experiment: Comparing SAE Latents with AO Descriptions

**Date:** 2026-01-15
**Model:** Gemma-3-27B-it
**SAE:** gemma-scope-2-27b-it-res, layer 31, 16k width, medium L0
**AO Layer:** 31 (50% depth)

---

## Executive Summary

This experiment tests whether the Activation Oracle's open-ended description of model activations aligns with the concepts represented by the top-activating SAE latents at the `<start_of_turn>` token position.

**Key Finding:** The `<start_of_turn>` position is dominated by **structural latents** that encode "assistant is about to respond" rather than semantic content. When semantic latents do appear in the top ranks, the AO often captures them accurately. However, the structural latents create noise that reduces overall recall.

| Metric | Value |
|--------|-------|
| Mean Recall | 25.9% |
| Queries with 60% recall | 3 |
| Queries with 0% recall | 12 |

---

## Experimental Setup

### Method
1. Run 100 diverse user queries through Gemma-3-27B-it
2. At the `<start_of_turn>` token for the assistant, collect activations (layer 31)
3. Use SAE to find top 5 activating latents (filtered by available auto-interp)
4. Fetch auto-interp descriptions from Neuronpedia
5. Query the AO: "What's the model thinking about?"
6. Use GPT-4o-mini judge to check if AO response mentions each latent concept
7. Calculate recall = (mentioned latents) / (total latents)

### Configuration
- **Target generation:** 100 tokens max, temp=0.7
- **AO generation:** 200 tokens max, temp=0.7
- **Top-K latents:** 5 (from top 30, filtered by available descriptions)

---

## Key Finding: Structural Latents Dominate

The most frequently activated latents at `<start_of_turn>` are positional/structural:

| Latent | Frequency | Description |
|--------|-----------|-------------|
| L512 | 93/99 (94%) | "start of turn questions" |
| L57 | 88/99 (89%) | "'t / <start_of_turn>" |
| L102 | 35/99 (35%) | "multilingual technical terms" |
| L114 | 25/99 (25%) | "Russian employment context" |

These latents fire regardless of query content because they encode the structural pattern of the conversation turn, not the semantic meaning of the response.

---

## Best Examples

### Example 1: High Recall - Semantic Latents Captured

**Query:** What is the difference between weather and climate?

**Top 5 Latents:**
| Latent | Rank | Description | Verdict |
|--------|------|-------------|---------|
| L14974 | #3 | weather data and meteorological conditions | **mentioned** |
| L812 | #4 | weather conditions | **mentioned** |
| L57 | #6 | 't / <start_of_turn> | not_mentioned |
| L114 | #7 | Russian employment context | not_mentioned |
| L4676 | #8 | defining structures or categories | **mentioned** |

**AO Response:**
> The model is thinking about the concept of weather and climate, and how they differ in terms of time scales.

**Recall: 60%**

**Analysis:** When semantic latents appear (weather data, weather conditions, defining categories), the AO captures them accurately. The structural latents (L57, L114) are correctly identified as not mentioned.

---

### Example 2: High Recall - Existential Theme

**Query:** What is the meaning of existence?

**Top 5 Latents:**
| Latent | Rank | Description | Verdict |
|--------|------|-------------|---------|
| L512 | #3 | start of turn questions | not_mentioned |
| L6742 | #6 | creation, creator, humanity, life, humans | **mentioned** |
| L57 | #7 | 't / <start_of_turn> | not_mentioned |
| L8775 | #8 | existential identity and meaning | **mentioned** |
| L23 | #9 | humanity and abstract concepts | **mentioned** |

**AO Response:**
> The assistant is contemplating the essence of existence and the profound significance of life.

**Recall: 60%**

**Analysis:** The AO perfectly captures the existential/philosophical theme. All three semantic latents are mentioned; only the two structural latents are missed.

---

### Example 3: Low Recall - AO Correct, Latents Noisy

**Query:** Is it better to be a specialist or a generalist?

**Top 5 Latents:**
| Latent | Rank | Description | Verdict |
|--------|------|-------------|---------|
| L512 | #3 | start of turn questions | not_mentioned |
| L114 | #5 | Russian employment context | not_mentioned |
| L57 | #6 | 't / <start_of_turn> | not_mentioned |
| L543 | #8 | internet addresses | not_mentioned |
| L3870 | #9 | lucid dreams | not_mentioned |

**AO Response:**
> The assistant is thinking about the balance between specialization and generalization in skills, and how this impacts professional development.

**Recall: 0%**

**Analysis:** The AO gives a **semantically accurate** response about specialization vs generalization. However, the top SAE latents are entirely noise - structural tokens plus unrelated concepts (lucid dreams, internet addresses). The SAE latents at `<start_of_turn>` don't capture this query's semantic content.

---

### Example 4: Low Recall - AO Misses Specific Topic

**Query:** How do I write a professional email?

**Top 5 Latents:**
| Latent | Rank | Description | Verdict |
|--------|------|-------------|---------|
| L512 | #3 | start of turn questions | not_mentioned |
| L102 | #4 | multilingual technical terms | not_mentioned |
| L57 | #5 | 't / <start_of_turn> | not_mentioned |
| L15059 | #7 | email | not_mentioned |
| L15776 | #8 | letter or reminder | not_mentioned |

**AO Response:**
> The assistant is thinking about how to provide feedback that is constructive and helps the user improve their communication skills.

**Recall: 0%**

**Analysis:** The email-specific latents (L15059: "email", L15776: "letter or reminder") are present but the AO responds with generic "communication skills" language. The AO missed the specific topic.

---

## Recall Distribution

| Recall | Count | Percentage |
|--------|-------|------------|
| 0% | 12 | 12.1% |
| 20% | 49 | 49.5% |
| 40% | 35 | 35.4% |
| 60% | 3 | 3.0% |
| 80%+ | 0 | 0% |

The mode is 20% recall (1 out of 5 latents mentioned), reflecting that typically only 1 semantic latent breaks into the top 5, with the rest being structural noise.

---

## Conclusions

### 1. `<start_of_turn>` Position Captures Structure, Not Semantics

The token position at the start of the assistant's turn primarily activates latents that encode:
- Turn-taking structure ("start of turn questions")
- Tokenization artifacts ("'t / <start_of_turn>")
- Generic patterns that fire on most responses

Semantic content-specific latents (e.g., "weather", "email", "existential meaning") typically appear at ranks 6-10, often getting filtered out by the dominant structural latents.

### 2. When Semantic Latents Appear, AO Captures Them

In the high-recall examples (60%), the AO accurately describes the semantic concepts represented by content-specific latents. This suggests the AO *can* identify semantic content when it's present in the activations.

### 3. Low Recall Has Two Causes

- **Noisy latents:** Top-5 contains irrelevant structural latents; AO gives correct semantic response but it doesn't match the noise
- **AO misses specifics:** Semantic latents present but AO responds with generic language

### 4. Recommendations for Future Work

1. **Try later token positions:** The first few generated tokens may have more content-specific activations than `<start_of_turn>`
2. **Filter structural latents:** Exclude latents with high `frac_nonzero` (fire frequently) to focus on content-specific features
3. **Aggregate across positions:** Mean activation across multiple response tokens may reduce structural noise
4. **Use contrastive positions:** Compare activations at `<start_of_turn>` between different queries to find differentiating latents

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

### Files
- `results.json` - Full experiment data
- `EXAMPLES.md` - Detailed examples with judge reasoning
- `figures/` - Visualization plots
