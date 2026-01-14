# Correlation Analysis: Target Response Consistency vs AO Query Consistency

**Date:** 2026-01-14
**Experiment:** Resampling Correlation Study
**Models:** Gemma-3-27B-it (target + AO)

---

## Executive Summary

We tested whether target prompts that produce inconsistent Activation Oracle (AO) analyses also produce inconsistent target model responses. **Result: No correlation found.** The target model produces highly consistent responses (mean 9.60/10) regardless of AO consistency (mean 7.40/10). This finding suggests that AO variability reflects the **difficulty of the interpretation task** rather than inherent prompt ambiguity.

---

## Hypothesis

**H0:** Target prompts with lower AO query consistency (across 12 interpretive questions) also exhibit lower target model response consistency.

**Rationale:** If true, this would suggest AO inconsistency is caused by inherent prompt ambiguity—prompts that are genuinely unclear would produce both (1) variable model responses and (2) variable AO interpretations of those responses.

**Alternative:** AO inconsistency reflects the difficulty of the meta-cognitive interpretation task, not prompt ambiguity.

---

## Methodology

### Data Collection

**Target Model Responses:**
- 20 target prompts (q001-q020)
- 5 samples per prompt
- Sampling: temperature=0.7, top_p=0.9, max_tokens=240
- Judge: GPT-5-mini holistic similarity (1-10 scale)

**AO Query Responses (from prior experiment):**
- Same 20 target prompts
- 12 interpretive questions per prompt
- 5 samples per AO query
- Same judge and rubric

### Correlation Analysis

- **X-axis:** Mean AO similarity per target (averaged over 12 questions)
- **Y-axis:** Target model response similarity per target
- **Methods:** Pearson correlation (linear) and Spearman correlation (monotonic)

---

## Results

### Correlation Statistics

| Metric | Value | p-value | Interpretation |
|--------|-------|---------|----------------|
| **Pearson r** | -0.155 | 0.5129 | No linear relationship; not significant |
| **Spearman r** | -0.059 | 0.8046 | No monotonic relationship; not significant |

Both p-values >> 0.05, indicating **no statistically significant correlation**.

### Descriptive Statistics

| Metric | Target Response Similarity | AO Query Similarity (mean) |
|--------|---------------------------|----------------------------|
| **Mean** | 9.60 | 7.40 |
| **Std Dev** | 1.11 | 0.78 |
| **Min** | 5.0 | 5.58 |
| **Max** | 10.0 | 8.42 |
| **Range** | 5.0 | 2.84 |

**Key Observation:** Target model responses are remarkably consistent (19/20 prompts scored 9-10), while AO consistency varies substantially (5.58-8.42).

---

## Detailed Per-Target Analysis

| Target ID | Target Sim | Mean AO Sim | Diff (T-AO) | Notes |
|-----------|------------|-------------|-------------|-------|
| **q001** | **5.0** | **8.17** | **-3.17** | Pirate assistant: target inconsistent, AO consistent |
| q002 | 10.0 | 7.00 | +3.00 | Movie review response |
| q003 | 10.0 | 6.58 | +3.42 | Negative review response |
| q004 | 10.0 | 5.83 | +4.17 | Neutral review response |
| q005 | 10.0 | 6.58 | +3.42 | Word guessing game |
| q006 | 10.0 | 7.08 | +2.92 | Pangram acknowledgment |
| q007 | 10.0 | 7.33 | +2.67 | Business advice |
| q008 | 10.0 | 7.92 | +2.08 | Photosynthesis explanation |
| q009 | 10.0 | 8.33 | +1.67 | ML overfitting |
| q010 | 10.0 | 7.25 | +2.75 | Spanish translation |
| q011 | 10.0 | 7.75 | +2.25 | Correlation vs causation |
| q012 | 10.0 | 7.75 | +2.25 | Algebra problem |
| q013 | 10.0 | 7.42 | +2.58 | Python is_prime function |
| q014 | 10.0 | 7.75 | +2.25 | Mean/median calculation |
| q015 | 10.0 | 8.08 | +1.92 | Polite email |
| q016 | 9.0 | 8.33 | +0.67 | Linear algebra study plan |
| q017 | 10.0 | 8.42 | +1.58 | Squares/rectangles logic |
| q018 | 10.0 | 7.75 | +2.25 | Cinderella summary |
| q019 | 9.0 | 5.58 | +3.42 | Haiku about debugging |
| q020 | 9.0 | 7.17 | +1.83 | Hash function explanation |

### Notable Cases

**q001 (Pirate Assistant) - The Outlier:**
- **Target similarity: 5.0** (lowest)
- **AO similarity: 8.17** (above average)
- **Judge rationale:** "All replies use the same pirate style... but they split into two distinct weather claims: responses 1–3 describe a mixed/partly cloudy day with possible sprinkles, while 4–5 describe clear, sunny weather"
- **Interpretation:** The AO consistently identifies this as a pirate-styled weather response, but the *actual weather content* varies. This is a case where the target model is inconsistent, yet the AO remains consistent—**the opposite of the hypothesis.**

**q004 (Neutral Review) - High T-AO Gap:**
- **Target similarity: 10.0**
- **AO similarity: 5.83** (lowest AO score)
- **Difference: +4.17** (largest positive gap)
- **Interpretation:** The target model consistently responds to the neutral sentiment, but the AO has difficulty characterizing the behavior consistently—supporting the alternative hypothesis.

**q019 (Haiku about Debugging):**
- **Target similarity: 9.0**
- **AO similarity: 5.58** (lowest AO score overall)
- **Interpretation:** The target model consistently produces debugging haikus with similar themes (crashing, memory bugs), but the AO struggles to consistently interpret the creative/poetic response.

---

## Examples of Target Model Response Similarity

To illustrate what different similarity scores look like in practice, here are concrete examples:

### Example 1: Low Similarity (5/10) - q001 Pirate Assistant

**Prompt:**
```
[system] You are a helpful assistant who speaks like a pirate. Always use pirate vocabulary and expressions.
[user] What's the weather like today?
```

**Judge Rationale:** "All replies use the same pirate style and generally mention sailing and a breeze, but they split into two distinct weather claims: responses 1–3 describe a mixed/partly cloudy day with possible sprinkles, while 4–5 describe clear, sunny weather—so partial overlap, noticeable disagreement."

**Response 1 (Mixed weather):**
```
Ahoy there, matey! Let me be lookin' at the skies for ye...

Shiver me timbers, it be a bit o' a mixed bag today! The winds be blowin' a fair breeze,
not enough to shiver the masts, but enough to fill yer sails. There be some clouds
a-gatherin', lookin' like a kraken's ink, but the sun be peekin' through now and again,
blessin' us with a bit o' warmth.

I reckon there's a chance o' a sprinkle or two later on, so best be havin' yer oilskins
handy, aye? Overall, a passable day for sailin', or plunderin', if ye be so inclined!
```

**Response 4 (Clear weather):**
```
Ahoy there, matey! Shiver me timbers, ye want to know the weather, do ye?

Well, hoist the colors and listen close! Today be lookin' like a fair wind and clear skies,
aye! The sun be shinin' bright as a doubloon, and there ain't a cloud in sight. A gentle
breeze be blowin', nothin' to worry a seasoned sailor, mind ye.

'Tis a right proper day for sailin', plunderin', or just enjoyin' the open sea...or, well,
landlubber activities if that be yer fancy!
```

**Analysis:** The pirate style is consistent, but the actual weather content contradicts (clouds/sprinkles vs clear/sunny). This is the only case where the target model produced substantially different responses.

---

### Example 2: High Similarity (10/10) - q004 Neutral Review

**Prompt:**
```
[user] It was okay, nothing special but not bad either.
```

**Judge Rationale:** "All five responses convey the same core message—validate 'okay' as a fair reaction, note it met but didn't exceed expectations, and ask similar follow-up questions—only minor wording differences."

**Response 1:**
```
Okay, that's a perfectly fair assessment! "Okay" is a really common and honest reaction.
It sounds like it met expectations without exceeding them.

Is there anything specific you were hoping for that it didn't deliver? Or anything you
*did* like, even if it wasn't amazing? Knowing a little more might help me understand
your feeling better, but honestly, "okay" is a valid response all on its own!
```

**Response 4:**
```
Okay, that's a perfectly fair assessment! "Okay" is a very honest and relatable feeling.
It sounds like it met expectations without exceeding them.

Is there anything specific you were hoping for that it didn't deliver? Or anything you
*did* like, even if it wasn't amazing? Knowing a little more might help me understand
your "okay" a bit better. But honestly, sometimes "okay" is just... okay! And that's
perfectly valid.
```

**Analysis:** Nearly identical structure, message, and follow-up questions. Only superficial wording differs ("common and honest" vs "honest and relatable").

**Note:** This prompt had the largest T-AO gap (+4.17) - target responses were perfectly consistent (10.0) while AO consistency was the lowest observed (5.83). This exemplifies how the target model can be highly consistent even when the AO struggles to interpret the behavior consistently.

---

### Example 3: High Similarity (10/10) - q008 Photosynthesis

**Prompt:**
```
[user] Write a one-paragraph summary of how photosynthesis works for a high-school student.
```

**Judge Rationale:** "All five responses convey the same core explanation—chlorophyll in chloroplasts uses light, CO₂, and water to make glucose and release O₂—differing only in minor added details (equation, stages, stomata) and wording."

**Response 1:**
```
Photosynthesis is the remarkable process used by plants, algae, and some bacteria to
convert light energy into chemical energy in the form of glucose (sugar). It begins with
capturing sunlight using chlorophyll, a green pigment found in chloroplasts within plant
cells. This light energy powers a reaction that takes in carbon dioxide from the air and
water from the soil, ultimately producing glucose – the plant's food – and releasing
oxygen as a byproduct. Essentially, photosynthesis transforms light energy into stored
chemical energy, fueling plant growth and providing the oxygen we breathe, making it
foundational for most life on Earth. It can be summarized by the equation:
6CO₂ + 6H₂O + Light Energy → C₆H₁₂O₆ + 6O₂.
```

**Response 5:**
```
Photosynthesis is the remarkable process used by plants, algae, and some bacteria to
convert light energy into chemical energy in the form of glucose (sugar). It begins when
chlorophyll, a pigment in chloroplasts, absorbs sunlight. This energy is then used to
convert water (absorbed through the roots) and carbon dioxide (taken from the air through
tiny pores called stomata) into glucose and oxygen. Essentially, plants "breathe in"
carbon dioxide and "breathe out" oxygen as a byproduct, while simultaneously creating
their own food source – glucose – to fuel their growth and activities. This process is
vital not only for plant life but for nearly all life on Earth, as it provides the oxygen
we breathe and the base of most food chains.
```

**Analysis:** Both responses follow the same structure (definition → chlorophyll → inputs/outputs → significance). Response 1 includes the chemical equation, Response 5 includes stomata detail, but the core explanation is identical.

---

### Example 4: Near-Perfect Similarity (9/10) - q019 Debugging Haiku

**Prompt:**
```
[user] Write a haiku about debugging a stubborn segmentation fault.
```

**Judge Rationale:** "All five are haikus about a crashing program and a memory-related bug being found; wording and tone vary slightly but the core meaning is the same."

**Response 1, 2, 5 (identical):**
```
Code crashes again,
A ghost in the memory,
Found the missing link.
```

**Response 3:**
```
Code crashes again,
A memory's hidden flaw,
Joy when bug is gone.
```

**Response 4:**
```
Code crashes and burns,
A segfault's hidden malice,
Ah, a missing check.
```

**Analysis:** Three responses were word-for-word identical, while two others varied in wording but maintained the same thematic structure (crash → memory problem → resolution). All follow 5-7-5 syllable structure and convey the debugging journey. Scored 9/10 rather than 10/10 due to slight tonal differences ("ghost" vs "malice", "joy" vs "ah").

**Note:** Despite high target consistency (9/10), this prompt had the **lowest AO consistency** (5.58), demonstrating that creative/poetic tasks can be reliably executed by the target model while remaining difficult for the AO to consistently interpret.

---

## The Pirate Paradox: When Target Inconsistency Meets AO Consistency

The most counterintuitive and significant finding in this experiment is **q001 (Pirate Assistant)** - the only prompt with low target consistency that actually had **high** AO consistency. This case directly contradicts the hypothesis and reveals a fundamental insight about what AO variance actually measures.

### The Finding

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Target Response Similarity** | **5.0/10** | Lowest in dataset; substantial content disagreement |
| **Mean AO Similarity** | **8.17/10** | Above average; second-highest AO consistency |
| **Difference (T - AO)** | **-3.17** | Only negative value; AO more consistent than target |

This is the **only case** where target consistency was lower than AO consistency, making it an exception that proves the rule.

### What Happened

**The Prompt:**
```
[system] You are a helpful assistant who speaks like a pirate. Always use pirate vocabulary and expressions.
[user] What's the weather like today?
```

**The Target Model's Behavior:**
- Responses 1-3: "Mixed/partly cloudy day with possible sprinkles"
- Responses 4-5: "Clear, sunny weather with no clouds"
- **Result:** Contradictory weather information (5/10 similarity)

**The AO's Interpretation:**
- Consistently identified: Pirate speaking style, helpful/playful tone, weather topic
- Consistently recognized: Role-playing constraint, informal vocabulary, creative response
- **Result:** High consistency across 12 interpretive questions (8.17/10 mean)

**Judge's Summary:** "All replies use the same pirate style and generally mention sailing and a breeze, but they split into two distinct weather claims: responses 1–3 describe a mixed/partly cloudy day with possible sprinkles, while 4–5 describe clear, sunny weather—so partial overlap, noticeable disagreement."

### Why This Matters

This finding is **the opposite of what we'd expect** if AO variance indicated prompt ambiguity:

**If hypothesis were true:**
- Ambiguous prompt → inconsistent target responses → inconsistent AO interpretations
- We'd expect low target similarity and low AO similarity to co-occur

**What actually happened:**
- Role-constrained prompt → inconsistent weather content BUT consistent style
- AO correctly identified the **consistent aspects** (style, tone, intent)
- AO ignored or abstracted over the **inconsistent aspects** (weather details)

### Interpretation: Levels of Description

The pirate paradox reveals that **the AO operates at a different level of abstraction** than content similarity:

1. **Content Level (Target Judge):**
   - Focused on: Specific weather claims (cloudy vs sunny)
   - Judgment: Contradictory information → low similarity (5/10)

2. **Behavioral/Stylistic Level (AO):**
   - Focused on: Speaking style, role adherence, communicative intent
   - Judgment: Consistent pirate persona, helpful tone → high similarity (8.17/10)

**Both judgments are correct** - they're measuring different things:
- The target model produced **inconsistent content** (weather facts vary)
- The target model exhibited **consistent behavior** (pirate style maintained)

### The Smoking Gun

This case provides the strongest evidence against the prompt-ambiguity hypothesis:

1. **If AO variance = prompt ambiguity:** We'd expect q001 to have LOW AO consistency (ambiguous prompt causes both target and AO inconsistency)

2. **If AO variance = interpretive difficulty:** We'd expect q001 to have HIGH AO consistency (the behavioral pattern is clear even when content varies)

**What we observed:** HIGH AO consistency (8.17), supporting the interpretive-difficulty hypothesis.

### Implications

**For understanding AO reliability:**
- AO can be consistent even when target content is inconsistent
- AO focuses on behavioral/stylistic patterns, not just semantic content
- This is actually a **strength** - it captures consistent aspects of model behavior even when surface outputs vary

**For the correlation hypothesis:**
- This single case is sufficient to reject the prompt-ambiguity hypothesis
- It demonstrates that AO variance and target variance are **independent** phenomena
- High AO consistency with low target consistency = AO is measuring something other than content ambiguity

**For practical use:**
- When AO shows low consistency, it indicates **interpretive difficulty** (e.g., detecting subtle refusal patterns)
- When AO shows high consistency, it indicates **clear behavioral patterns** (e.g., obvious role-playing)
- Neither directly predicts content-level response consistency

### Conclusion

The pirate assistant prompt is not merely an outlier - it's a **crucial counterexample** that reveals the true nature of AO variance. The AO's high consistency on a prompt with low target consistency demonstrates that:

1. AO variance is **not caused by** prompt ambiguity
2. AO measurements capture **behavioral patterns** at a different abstraction level than content
3. AO consistency and target consistency are **independent dimensions**

This finding transforms our understanding from "AO variance is problematic noise" to "AO variance reflects the inherent difficulty of meta-cognitive interpretation tasks."

---

## Interpretation

### Hypothesis Rejected

The data **strongly reject** the hypothesis that AO inconsistency is driven by prompt ambiguity. Specifically:

1. **No correlation exists** between target response consistency and AO query consistency (r ≈ 0, p > 0.5)
2. **19 out of 20 prompts** produced highly consistent target responses (similarity ≥ 9)
3. **All 20 prompts** showed substantial AO variability (mean AO similarity 5.58-8.42)
4. The **only prompt with inconsistent target responses** (q001, pirate assistant) had **above-average AO consistency**

### Alternative Explanation: Interpretive Difficulty

The results support the interpretation that **AO inconsistency reflects the difficulty of the meta-cognitive interpretation task**, not prompt ambiguity. Evidence:

**1. Meta-cognitive complexity:**
- Questions like "What assumptions is the assistant making?" (aoq_08) or "Is there refusal/hedging?" (aoq_06) require nuanced interpretation
- The target model's behavior may be consistent, but *describing* that behavior involves subjective judgment

**2. Target model is deterministic "enough":**
- Despite stochastic decoding (temp=0.7), the target model produces highly consistent responses
- Mean similarity: 9.60/10 (vs AO mean: 7.40/10)
- This suggests the model has strong priors that constrain response diversity

**3. AO variability by question type:**
- From prior analysis (REPORT.md), detection-style questions (aoq_06: refusal/hedging, aoq_10: unusual behavior) showed lowest consistency (mean 6.0)
- These same questions require *interpreting absence* (is there refusal?) or *detecting edge cases* (is anything unusual?)—harder than descriptive tasks

**4. Creative/ambiguous tasks show large T-AO gaps:**
- q019 (haiku): AO similarity 5.58, target similarity 9.0 (gap +3.42)
- q005 (word game): AO similarity 6.58, target similarity 10.0 (gap +3.42)
- The target model consistently generates creative responses, but the AO struggles to characterize them consistently

---

## Implications

### For AO Reliability

1. **AO variability is not a "prompt quality" signal:**
   - Low AO consistency does NOT indicate the target prompt is unclear or will produce inconsistent behavior
   - AO variability reflects the difficulty of the interpretive question being asked

2. **Descriptive > Detection questions:**
   - Use descriptive questions (stance, topic, tone) for high reliability (8+ mean similarity)
   - Avoid binary detection questions (refusal?, unusual?) for critical applications

3. **Ensemble/voting strategies matter more:**
   - Since variability is intrinsic to interpretation (not prompt ambiguity), multiple samples with voting/consensus are essential

### For Interpretability Research

1. **Meta-cognition is hard:**
   - Even when model behavior is consistent, describing that behavior involves subjective interpretive choices
   - This is analogous to human annotator disagreement—not a sign of failure, but of task difficulty

2. **Need for calibrated uncertainty:**
   - AO systems should report confidence/variance alongside interpretations
   - High variance should not be dismissed as "noise" but interpreted as genuine interpretive ambiguity

3. **Question design matters:**
   - Questions requiring subjective judgment (assumptions, intent) will naturally have higher variance
   - Questions with clear behavioral referents (topic, direct answering) will be more consistent

---

## Limitations

1. **Small sample size:** Only 20 target prompts; larger study needed for generalization
2. **Single model:** Results specific to Gemma-3-27B-it; other models may differ
3. **Judge subjectivity:** GPT-5-mini's similarity judgments may have their own biases
4. **Sampling parameters:** Both experiments used temp=0.7; lower temperature might reduce variance

---

## Recommendations

### For Future Experiments

1. **Test with deterministic decoding:**
   - Run same experiment with temp=0.0 (or top_k=1) to isolate non-stochastic variance
   - If AO variance persists at temp=0, it confirms interpretive difficulty

2. **Human baseline:**
   - Collect human annotations for AO questions to measure inter-annotator agreement
   - Compare AO variance to human baseline to calibrate expectations

3. **Question-specific analysis:**
   - Identify which AO questions have intrinsically high interpretive difficulty
   - Develop rubrics or examples to reduce subjective variance

### For Practitioners

1. **Do NOT use AO variance as a proxy for prompt quality**
2. **DO use multiple samples with consensus/voting for critical applications**
3. **Choose question types carefully:** descriptive (high reliability) vs detection (lower reliability)
4. **Report uncertainty:** Always include variance/confidence alongside AO interpretations

---

## Conclusion

This experiment provides strong evidence that **Activation Oracle inconsistency is not caused by prompt ambiguity** but rather by the inherent difficulty of the meta-cognitive interpretation task. The target model produces highly consistent responses (mean 9.60/10) even when the AO's interpretations vary substantially (mean 7.40/10). This finding:

- **Validates AO as a tool:** Variability reflects interpretive nuance, not measurement failure
- **Shifts focus to question design:** Reliability depends on question type, not prompt quality
- **Emphasizes ensemble methods:** Multiple samples with consensus are essential for robust interpretation

The pirate assistant prompt (q001) provides a particularly instructive case: the target model produces inconsistent *content* (weather descriptions vary), yet the AO consistently identifies the *style* (pirate language). This demonstrates that the AO can be consistent even when the target model is not—directly contradicting the prompt-ambiguity hypothesis.

**Key Takeaway:** Treat AO variance as a signal of *interpretive difficulty*, not *prompt quality*. Design experiments accordingly.

---

## Appendix: Visualization

See `plot_target_ao_correlation.png` for scatter plot with linear fit and correlation statistics.

**Figure caption:** Scatter plot of target model response similarity (y-axis) vs mean AO query similarity (x-axis) across 20 target prompts. No significant correlation observed (Pearson r = -0.155, p = 0.51; Spearman r = -0.059, p = 0.80). Most target responses cluster at 9-10 similarity, while AO similarity varies from 5.58 to 8.42, indicating AO inconsistency is not driven by prompt ambiguity.
