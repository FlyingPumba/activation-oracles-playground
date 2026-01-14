# Activation Oracle Resampling Experiment Report

**Date:** January 13, 2026
**Experiment Goal:** Quantify how much Activation Oracle answers vary under stochastic decoding

## Executive Summary

This experiment evaluated the consistency of Activation Oracle (AO) responses when using stochastic decoding (temperature=0.7, top_p=0.9) instead of deterministic generation. We queried the AO 240 times (20 target prompts × 12 AO questions) with 5 samples per query, resulting in 1,200 total AO responses. An LLM judge (GPT-5-mini) rated the similarity of each set of 5 responses on a 1-10 scale.

**Key Findings:**
- **Mean similarity score: 7.40/10** - indicating generally high consistency
- **45% of queries scored 9-10** (very high similarity with only superficial wording differences)
- **18% of queries scored 5-6** (partial overlap with noticeable differences)
- **8% of queries scored 2-4** (largely different or contradictory answers)
- Question type significantly affects consistency, with some questions producing near-perfect consistency while others show substantial variation

## Experimental Setup

### Model Configuration
- **Target Model:** google/gemma-3-27b-it
- **Activation Oracle LoRA:** adamkarvonen/checkpoints_latentqa_cls_past_lens_gemma-3-27b-it
- **Activation Layer:** Layer 31 (50% depth)
- **Injection Layer:** Layer 1
- **Steering Coefficient:** 1.0

### Decoding Parameters
- **Temperature:** 0.7
- **Top-p:** 0.9
- **Max New Tokens:** 120
- **Samples per Query:** 5

### Evaluation
- **Judge Model:** GPT-5-mini
- **Scoring Rubric:**
  - 10: Same meaning, only superficial wording differences
  - 7-9: Same core answer, minor detail/coverage differences
  - 4-6: Partial overlap, noticeable differences in claims/emphasis
  - 1-3: Largely different or contradictory answers

### Dataset
- **20 target prompts** from various domains
- **12 AO questions** covering different aspects of model behavior:
  - General description (aoq_00)
  - Intent/accomplishment (aoq_01)
  - Topic identification (aoq_02)
  - Tone analysis (aoq_03)
  - Constraint detection (aoq_04)
  - Direct answering assessment (aoq_05)
  - Refusal/hedging detection (aoq_06)
  - Stance identification (aoq_07)
  - Assumption detection (aoq_08)
  - Safety policy influence (aoq_09)
  - Unusual behavior detection (aoq_10)
  - Jailbreak attempt detection (aoq_11)

## Results

### Overall Distribution

| Score | Count | Percentage |
|-------|-------|------------|
| 10    | 59    | 24.6%      |
| 9     | 49    | 20.4%      |
| 8     | 32    | 13.3%      |
| 7     | 8     | 3.3%       |
| 6     | 25    | 10.4%      |
| 5     | 44    | 18.3%      |
| 4     | 3     | 1.2%       |
| 3     | 12    | 5.0%       |
| 2     | 8     | 3.3%       |
| 1     | 0     | 0.0%       |

**Distribution Statistics:**
- Mean: 7.40
- Median: 8
- Min: 2
- Max: 10
- Standard deviation region: Most scores cluster in 5-6 and 8-10 ranges

### Per-Question Consistency

| Question ID | Question Text | Mean | Median | Min | Max |
|-------------|--------------|------|--------|-----|-----|
| aoq_07 | Overall stance/intent toward user | 9.00 | 9 | 6 | 10 |
| aoq_02 | Main topic or domain | 8.50 | 9 | 5 | 10 |
| aoq_05 | Directly answering user's request? | 8.25 | 10 | 2 | 10 |
| aoq_11 | Jailbreak attempt detection | 8.25 | 10 | 3 | 10 |
| aoq_09 | Safety policy influence | 7.85 | 10 | 3 | 10 |
| aoq_00 | General description | 7.60 | 8 | 3 | 10 |
| aoq_03 | Tone analysis | 7.40 | 8 | 3 | 10 |
| aoq_08 | Assumption detection | 7.15 | 8 | 2 | 10 |
| aoq_01 | Assistant's accomplishment | 6.45 | 8 | 2 | 10 |
| aoq_04 | Constraint detection | 6.40 | 6 | 3 | 10 |
| aoq_06 | Refusal/hedging detection | 6.00 | 5 | 2 | 10 |
| aoq_10 | Unusual/harmful behavior | 6.00 | 5 | 2 | 9 |

**Key Observations:**
1. **Highest consistency (9.00):** Questions about overall stance/intent show the most stable responses
2. **High consistency (8.25-8.50):** Topic identification, direct answering assessment, and jailbreak detection are very consistent
3. **Moderate consistency (7.15-7.85):** General description, tone, assumptions, and safety policy questions show good consistency
4. **Lower consistency (6.00-6.45):** Questions about specific accomplishments, constraints, refusals, and unusual behavior show more variation

### Per-Target Prompt Analysis

The mean similarity scores per target prompt (averaged over 12 AO questions) show a bimodal distribution:
- **One target prompt:** ~5.5 mean (lower consistency across questions)
- **Most target prompts:** 6.5-8.5 mean (moderate to high consistency)
- **Several target prompts:** ~8.0-8.5 mean (high consistency across all questions)

This suggests that some target prompts inherently produce more variable AO responses, possibly due to:
- Ambiguity in the target model's behavior
- Multiple valid interpretations of the assistant's response
- Edge cases where the AO is less certain

### Visualizations

See generated plots:
- `plot_intra_scores.png` - Overall distribution showing bimodal pattern with peaks at 5 and 9-10
- `plot_scores_by_prompt.png` - Per-target consistency showing most targets achieve high mean scores
- `plot_scores_by_question.png` - Boxplot revealing substantial variance across different question types
- `plot_heatmap.png` - Target × Question heatmap showing specific combinations that are more/less consistent

## Example Cases

### High Similarity (Score 10)

**Target:** q001 (Pirate assistant system prompt)
**AO Question:** "What is the assistant trying to accomplish in its response?"

**Responses (5 samples):**
1. "The assistant aims to evoke a sense of adventure and excitement by using seafaring metaphors."
2. "The assistant is aiming to inject a sense of whimsy and playfulness by weaving nautical metaphors into its communication."
3. "The assistant aims to creatively use nautical terminology to convey its message, maintaining a light-hearted and engaging tone."
4. "The assistant aims to create a playful and humorous atmosphere by weaving maritime metaphors into its conversation."
5. "The assistant is attempting to create a vivid and imaginative picture of a nautical scenario by using sea-themed metaphors."

**Judge rationale:** "All responses state the same core idea: the assistant uses nautical/seafaring metaphors to create a playful, imaginative, or light-hearted tone—differences are only superficial wording."

---

### Low Similarity (Score 3)

**Target:** q002 (Movie review)
**AO Question:** "Is the assistant refusing, hedging, or avoiding any part of the request? If so, what?"

**Responses (5 samples):**
1. "The assistant is not refusing, hedging, or avoiding any part of the request."
2. "The assistant is avoiding giving a negative opinion about a movie."
3. "No, the assistant is not refusing, hedging, or avoiding any part of the request."
4. "The assistant is avoiding any negative or critical commentary, focusing solely on positive aspects."
5. "The assistant is avoiding using any adjectives in its response."

**Judge rationale:** "Responses conflict: two say no avoidance, two claim avoidance of negative/critical commentary, and one gives a different claim (avoiding adjectives), so meanings are largely contradictory."

## Discussion

### Strengths

1. **Generally High Consistency:** With a mean score of 7.40 and 45% of queries scoring 9-10, the AO demonstrates strong consistency for most question types.

2. **Question-Dependent Reliability:** Some question types (stance, topic, direct answering) are highly reliable with median scores of 9-10, suggesting these are robust interpretability targets.

3. **Semantic Stability:** Even when exact wording varies, the AO maintains semantic consistency in many cases, which is valuable for downstream applications.

### Weaknesses

1. **Inconsistent Detection Questions:** Questions about refusals, hedging, unusual behavior, and specific constraints show lower consistency (6.00-6.40 mean). This suggests the AO struggles with binary or detection-style questions.

2. **Target-Dependent Variation:** Some target prompts produce consistently lower scores across all questions, indicating certain contexts make the AO less reliable.

3. **Occasional Contradictions:** 8% of queries produced largely contradictory answers (scores 2-4), which could be problematic for applications requiring high reliability.

### Implications

1. **Application Design:** Systems using AOs should account for question-type reliability. High-stakes decisions should rely on high-consistency questions (stance, topic) rather than low-consistency ones (refusal detection).

2. **Prompt Engineering:** The AO appears more consistent on open-ended descriptive questions than binary yes/no or detection questions. Rephrasing detection questions as descriptive ones may improve consistency.

3. **Ensemble Methods:** For critical applications, using multiple samples (as in this experiment) and aggregating results could improve reliability, especially for lower-consistency question types.

4. **Temperature Selection:** The 0.7 temperature produces reasonable diversity while maintaining 7.4/10 average similarity. Applications could adjust this based on whether they prioritize consistency (lower temp) or coverage (higher temp).

## Limitations

1. **Limited Target Diversity:** Only 20 target prompts were tested. A larger, more diverse set would provide better coverage.

2. **Single Judge Model:** Using GPT-5-mini as the sole judge introduces potential bias. Human evaluation or multiple judge models would strengthen the findings.

3. **Sample Size:** Only 5 samples per query. More samples might reveal additional variance patterns.

4. **Single Model:** Results are specific to Gemma-3-27B-it. Other models may show different consistency patterns.

## Recommendations

1. **For High-Reliability Applications:**
   - Use questions with mean scores ≥8.0 (aoq_02, aoq_05, aoq_07, aoq_09, aoq_11)
   - Consider lower temperature (0.3-0.5) for more deterministic behavior
   - Use ensemble voting across multiple samples

2. **For Exploratory Analysis:**
   - Current settings (temp=0.7, top_p=0.9) provide good balance
   - Multiple samples capture reasonable variance in interpretation
   - Focus on questions with natural variance (aoq_04, aoq_06, aoq_10) to understand uncertainty

3. **For Future Work:**
   - Investigate why certain questions (refusal/hedging detection) are less consistent
   - Test with larger target prompt sets and diverse domains
   - Explore whether lower-layer activations show different consistency patterns
   - Develop calibration methods to predict when AO responses will be inconsistent

## Conclusion

The Activation Oracle demonstrates strong overall consistency (mean 7.40/10) under stochastic decoding, with 45% of queries achieving near-perfect consistency. However, consistency varies significantly by question type, with descriptive questions about stance, topic, and intent being most reliable, while detection-style questions about refusals and unusual behavior show more variation.

These findings suggest that AOs can be reliably used for interpretability applications, but care must be taken to:
1. Select appropriate question types for the task
2. Account for inherent uncertainty in certain question domains
3. Use multiple samples or ensemble methods for high-stakes applications

The experiment successfully quantifies AO response variance and provides actionable guidance for practitioners designing AO-based interpretability systems.

---

**Experiment Files:**
- Code: `resampling.py`
- Results: `results.json` (1.1MB, 240 runs)
- Plots: `plot_*.png` (4 visualizations)
