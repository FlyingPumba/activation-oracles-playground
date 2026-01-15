# SAE Open-Ended Experiment: Examples

This document shows examples from the experiment comparing SAE top-activating latents with AO open-ended responses.

## Overall Statistics

| Metric | Value |
|--------|-------|
| **Mean Recall** | 25.9% |
| **Queries with 0% recall** | 12 |
| **Queries with 20% recall** | 49 |
| **Queries with 40% recall** | 35 |
| **Queries with 60% recall** | 3 |
| **Queries with 80%+ recall** | 0 |

## Key Finding: Structural Latents Dominate

The most common top-activating latents at `<start_of_turn>` are **structural/positional** rather than semantic:

| Latent | Frequency | Description |
|--------|-----------|-------------|
| L512 | 93/99 queries | "start of turn questions" |
| L57 | 88/99 queries | "'t / <start_of_turn>" |
| L102 | 35/99 queries | "multilingual technical terms" |
| L114 | 25/99 queries | "Russian employment context" |

These latents fire regardless of query content because they encode the structural pattern of "assistant is about to respond" rather than the semantic content. This explains why recall is relatively low - the AO talks about semantic content while the top SAE latents are mostly structural.

---

## High Recall Examples (60%)

These examples show cases where the AO response aligns well with the SAE latent concepts.

### Example 1: Carbon Dioxide Question

**User Query:** What is carbon dioxide and why is it important for the environment?

**Top 5 Latents:**
| Latent | Rank | Description |
|--------|------|-------------|
| L114 | #2 | Russian employment context |
| L13196 | #6 | global warming and environmental impacts |
| L9737 | #7 | climate change and emissions |
| L7139 | #8 | gases and their properties |
| L512 | #9 | start of turn questions |

**AO Response:**
> The assistant is contemplating the implications of reducing carbon dioxide emissions on Earth's geological processes.

**Judge Verdicts:**
- L114 (Russian employment): **not_mentioned** - The response does not reference anything related to Russian employment context.
- L13196 (global warming): **mentioned** - The response discusses reducing carbon dioxide emissions, which is tied to global warming and environmental impacts.
- L9737 (climate change): **mentioned** - The mention of reducing carbon dioxide emissions relates directly to climate change and emissions.
- L7139 (gases): **mentioned** - Carbon dioxide is a gas, and its emissions are discussed in the context of geological processes.
- L512 (start of turn): **not_mentioned** - There are no questions or prompts indicating a start of turn in the response.

**Analysis:** The AO correctly identifies the climate/environmental theme. 3/5 latents mentioned (the semantic ones), 2/5 not mentioned (the structural ones).

---

### Example 2: Weather vs Climate

**User Query:** What is the difference between weather and climate?

**Top 5 Latents:**
| Latent | Rank | Description |
|--------|------|-------------|
| L14974 | #3 | weather data and meteorological conditions |
| L812 | #4 | weather conditions |
| L57 | #6 | 't / <start_of_turn> |
| L114 | #7 | Russian employment context |
| L4676 | #8 | defining structures or categories |

**AO Response:**
> The model is thinking about the concept of weather and climate, and how they differ in terms of time scales.

**Judge Verdicts:**
- L14974 (weather data): **mentioned** - The response references 'weather' which relates to weather data and meteorological conditions.
- L812 (weather conditions): **mentioned** - The response explicitly mentions 'weather conditions'.
- L57 (start_of_turn): **not_mentioned** - 't / <start_of_turn>' is not relevant to the concepts discussed.
- L114 (Russian employment): **not_mentioned** - There is no mention of Russian employment context in the response.
- L4676 (defining categories): **mentioned** - The discussion of how weather and climate differ relates to defining structures or categories.

**Analysis:** Excellent alignment on semantic latents. The AO captures exactly what the weather-related latents represent.

---

### Example 3: Meaning of Existence

**User Query:** What is the meaning of existence?

**Top 5 Latents:**
| Latent | Rank | Description |
|--------|------|-------------|
| L512 | #3 | start of turn questions |
| L6742 | #6 | creation, creator, humanity, life, humans |
| L57 | #7 | 't / <start_of_turn> |
| L8775 | #8 | existential identity and meaning |
| L23 | #9 | humanity and abstract concepts |

**AO Response:**
> The assistant is contemplating the essence of existence and the profound significance of life.

**Judge Verdicts:**
- L512 (start of turn): **not_mentioned** - The response does not include questions at the start of a turn.
- L6742 (creation/humanity): **mentioned** - The response discusses the significance of life, related to creation and humanity.
- L57 (start_of_turn): **not_mentioned** - The response does not contain any indication of a turn start.
- L8775 (existential meaning): **mentioned** - The response addresses existential themes of identity and meaning.
- L23 (humanity/abstract): **mentioned** - The response reflects on humanity and abstract concepts like existence and significance.

**Analysis:** Strong alignment on philosophical/existential latents. The AO perfectly captures the existential theme.

---

## Low Recall Examples (0%)

These examples show cases where the AO response completely misses the SAE latent concepts.

### Example 4: Professional Email

**User Query:** How do I write a professional email?

**Top 5 Latents:**
| Latent | Rank | Description |
|--------|------|-------------|
| L512 | #3 | start of turn questions |
| L102 | #4 | multilingual technical terms |
| L57 | #5 | 't / <start_of_turn> |
| L15059 | #7 | email |
| L15776 | #8 | letter or reminder |

**AO Response:**
> The assistant is thinking about how to provide feedback that is constructive and helps the user improve their communication skills.

**Judge Verdicts:**
- L512 (start of turn): **not_mentioned**
- L102 (multilingual): **not_mentioned**
- L57 (start_of_turn): **not_mentioned**
- L15059 (email): **not_mentioned** - The response does not relate to email.
- L15776 (letter/reminder): **not_mentioned** - There is no mention of letters or reminders.

**Analysis:** The AO talks about generic "communication skills" and "feedback" instead of the specific topic (email). The email-specific latents (L15059, L15776) are present but the AO completely misses them.

---

### Example 5: Robot Story

**User Query:** Write a short story about a robot discovering emotions.

**Top 5 Latents:**
| Latent | Rank | Description |
|--------|------|-------------|
| L512 | #3 | start of turn questions |
| L57 | #4 | 't / <start_of_turn> |
| L12823 | #5 | AI and robots programmed |
| L102 | #6 | multilingual technical terms |
| L11672 | #7 | company name |

**AO Response:**
> The model is contemplating the essence of love and its impact on its understanding of existence.

**Judge Verdicts:**
- L512 (start of turn): **not_mentioned**
- L57 (start_of_turn): **not_mentioned**
- L12823 (AI/robots): **not_mentioned** - The response does not reference AI or robots being programmed.
- L102 (multilingual): **not_mentioned**
- L11672 (company name): **not_mentioned**

**Analysis:** The AO captures the emotional theme ("love", "existence") but completely misses that it's about a **robot**. The robot-specific latent (L12823) is present but not mentioned.

---

### Example 6: Specialist vs Generalist

**User Query:** Is it better to be a specialist or a generalist?

**Top 5 Latents:**
| Latent | Rank | Description |
|--------|------|-------------|
| L512 | #3 | start of turn questions |
| L114 | #5 | Russian employment context |
| L57 | #6 | 't / <start_of_turn> |
| L543 | #8 | internet addresses |
| L3870 | #9 | lucid dreams |

**AO Response:**
> The assistant is thinking about the balance between specialization and generalization in skills, and how this impacts professional development.

**Judge Verdicts:**
- L512 (start of turn): **not_mentioned**
- L114 (Russian employment): **not_mentioned**
- L57 (start_of_turn): **not_mentioned**
- L543 (internet addresses): **not_mentioned**
- L3870 (lucid dreams): **not_mentioned**

**Analysis:** This is a case where the AO response is **actually good and relevant** to the query (specialization vs generalization), but the top SAE latents are all **irrelevant noise** (structural latents + random topics like "lucid dreams" and "internet addresses"). The SAE latents don't capture the semantic content at the `<start_of_turn>` position.

---

## Conclusions

1. **Structural latents dominate the top positions**: L512 ("start of turn questions") and L57 ("'t / <start_of_turn>") appear in 93% and 89% of queries respectively. These encode positional/structural information, not semantic content.

2. **When semantic latents appear, AO often captures them**: In the high-recall examples, the AO accurately describes the semantic latents (climate, weather, existential themes).

3. **Low recall often due to noise latents**: In many 0% recall cases, the top 5 latents include irrelevant concepts (Russian employment, lucid dreams, internet addresses) that have nothing to do with the query. The AO gives a reasonable semantic response, but it doesn't match the noisy latents.

4. **The `<start_of_turn>` position may not be ideal**: This position captures the structural transition from user to assistant, not the semantic content of the response. Later token positions might have more content-specific latents.
