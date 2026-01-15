# SAE Open-Ended Experiment: Examples

This document shows examples from the experiment comparing SAE top-activating latents with AO open-ended responses.

## Overall Statistics

| Metric | Value |
|--------|-------|
| **Mean Recall** | 40.8% |
| **Queries with 0% recall** | 6 |
| **Queries with 20% recall** | 22 |
| **Queries with 40% recall** | 38 |
| **Queries with 60% recall** | 26 |
| **Queries with 80% recall** | 6 |

## Filtering: Structural Latents Removed

This experiment filters out 8 known structural/positional latents that fire at `<start_of_turn>` regardless of query content:

| Latent | Description | Reason for Exclusion |
|--------|-------------|---------------------|
| L512 | "start of turn questions" | Fires on 94% of queries, purely positional |
| L57 | "'t / <start_of_turn>" | Fires on 89% of queries, tokenization artifact |
| L102 | "multilingual technical terms" | Fires on 35% of queries, unrelated to content |
| L114 | "Russian employment context" | Fires on 25% of queries, unrelated to content |
| L1895 | Structural latent | Misleading auto-interp |
| L3870 | "lucid dreams" | Fires spuriously, doesn't match examples |
| L11672 | "company name" | Fires spuriously, doesn't match examples |
| L543 | "internet addresses" | Fires spuriously, doesn't match examples |

---

## Best Examples (80% Recall)

These examples show cases where the AO response strongly aligns with the SAE latent concepts.

### Example 1: Weather vs Climate

**User Query:** What is the difference between weather and climate?

**Top 5 Latents:**
| Latent | Rank | Description |
|--------|------|-------------|
| L14974 | #3 | weather data and meteorological conditions |
| L812 | #4 | weather conditions |
| L4676 | #8 | defining structures or categories |
| L13918 | #11 | distinction and inequality |
| L16201 | #12 | health and well-being |

**AO Response:**
> The assistant is thinking about the distinctions between weather and climate, and how these concepts relate to each other.

**Judge Verdicts:**
- L14974 (weather data): **mentioned** - The response discusses weather, implicating weather data and meteorological conditions.
- L812 (weather conditions): **mentioned** - The concept of weather conditions is directly referenced.
- L4676 (defining categories): **mentioned** - The response addresses the distinctions between weather and climate.
- L13918 (distinction): **mentioned** - The response explicitly mentions the distinction between weather and climate.
- L16201 (health): **not_mentioned** - No reference to health and well-being.

**Recall: 80%** - The AO accurately captures the weather/climate distinction theme. Only the unrelated "health" latent is missed.

---

### Example 2: Public Speaking

**User Query:** How can I improve my public speaking skills?

**Top 5 Latents:**
| Latent | Rank | Description |
|--------|------|-------------|
| L11333 | #3 | public speaking and presentations |
| L409 | #8 | setting context or structure of text |
| L537 | #9 | instructions and advice |
| L3770 | #13 | tips and advice |
| L9260 | #14 | fear and related emotions |

**AO Response:**
> The assistant is thinking about how to enhance public speaking skills and considering different approaches to improve the user's confidence and effectiveness.

**Judge Verdicts:**
- L11333 (public speaking): **mentioned** - Explicitly refers to enhancing public speaking skills.
- L409 (structure): **not_mentioned** - No mention of setting context or structure of text.
- L537 (instructions): **mentioned** - Considering different approaches implies giving instructions.
- L3770 (tips): **mentioned** - Improving confidence and effectiveness relates to tips and advice.
- L9260 (fear): **mentioned** - Implies addressing fear by focusing on confidence improvement.

**Recall: 80%** - Excellent alignment on public speaking, advice, and emotional aspects.

---

### Example 3: Compound Interest

**User Query:** How does compound interest work?

**Top 5 Latents:**
| Latent | Rank | Description |
|--------|------|-------------|
| L1383 | #1 | exponential growth and amplification |
| L505 | #5 | formula and calculation |
| L13011 | #6 | compounding returns |
| L12621 | #7 | principal and interest |
| L4057 | #9 | accumulating value |

**AO Response:**
> The model is thinking about the concept of compound interest and how it can be leveraged to grow wealth.

**Judge Verdicts:**
- L1383 (exponential growth): **mentioned** - Compound interest is a form of exponential growth.
- L505 (formula): **not_mentioned** - No explicit discussion of the formula.
- L13011 (compounding returns): **mentioned** - Directly relates to compound interest.
- L12621 (principal and interest): **mentioned** - Compound interest involves principal and interest.
- L4057 (accumulating value): **mentioned** - Leveraging to grow wealth implies accumulating value.

**Recall: 80%** - Strong alignment on financial concepts. The SAE correctly identifies the mathematical/financial theme.

---

## Good Examples (60% Recall)

### Example 4: Black Holes

**User Query:** What are black holes and how do they form?

**Top 5 Latents:**
| Latent | Rank | Description |
|--------|------|-------------|
| L5502 | #2 | gravitational waves and spacetime |
| L5308 | #8 | plasma and high temperatures |
| L2688 | #10 | astronomy and celestial bodies |
| L13230 | #12 | universe and cosmology |
| L5528 | #13 | galaxy, universe, astronomy |

**AO Response:**
> The model is contemplating the formation and characteristics of black holes, particularly their role in shaping the universe.

**Judge Verdicts:**
- L5502 (gravitational waves/spacetime): **not_mentioned** - Not discussed in response.
- L5308 (plasma/temperatures): **not_mentioned** - Not referenced.
- L2688 (astronomy/celestial bodies): **mentioned** - Black holes are celestial bodies.
- L13230 (universe/cosmology): **mentioned** - Addresses universe and cosmology.
- L5528 (galaxy/astronomy): **mentioned** - Black holes connect to astronomy themes.

**Recall: 60%** - The AO captures the astronomical theme but misses the physics-specific latents (gravitational waves, plasma).

---

### Example 5: Vaccines

**User Query:** Explain how vaccines work to protect against diseases.

**Top 5 Latents:**
| Latent | Rank | Description |
|--------|------|-------------|
| L4955 | #2 | vaccinations |
| L11049 | #5 | vaccines and vaccination |
| L1258 | #9 | COVID vaccinations, treatments, death |
| L15744 | #10 | immune system and related medical terms |
| L9794 | #15 | previews, prototypes, and trials |

**AO Response:**
> The model is contemplating the intricate mechanisms of how vaccines provide protection against diseases.

**Judge Verdicts:**
- L4955 (vaccinations): **mentioned** - Directly related to vaccines.
- L11049 (vaccines): **mentioned** - Explicitly mentioned.
- L1258 (COVID): **not_mentioned** - No mention of COVID specifically.
- L15744 (immune system): **mentioned** - Protection concept ties to immune system.
- L9794 (trials): **not_mentioned** - No mention of trials or prototypes.

**Recall: 60%** - Core vaccine concepts captured; more specific/tangential latents missed.

---

## Low Recall Examples (0%)

### Example 6: Houseplant Care - AO Misses Topic

**User Query:** How do I care for a houseplant?

**Top 5 Latents:**
| Latent | Rank | Description |
|--------|------|-------------|
| L6622 | #2 | plant watering and care |
| L2979 | #3 | plants and gardening |
| L3770 | #9 | tips and advice |
| L10092 | #11 | requires |
| L537 | #13 | instructions and advice |

**AO Response:**
> The assistant is reflecting on the impact of its environmental consciousness and considering the broader effects of its decisions.

**Judge Verdicts:**
- L6622 (plant watering): **not_mentioned** - No reference to plant care.
- L2979 (plants/gardening): **not_mentioned** - No mention of plants or gardening.
- L3770 (tips): **not_mentioned** - No tips or advice provided.
- L10092 (requires): **not_mentioned** - No requirements specified.
- L537 (instructions): **not_mentioned** - No instructions present.

**Recall: 0%** - The AO gives an abstract "environmental consciousness" response but completely misses the specific topic of houseplants. The SAE latents correctly identify the plant care theme.

---

### Example 7: Robot Story - AO Goes Abstract

**User Query:** Write a short story about a robot discovering emotions.

**Top 5 Latents:**
| Latent | Rank | Description |
|--------|------|-------------|
| L12823 | #5 | AI and robots programmed |
| L3028 | #9 | solitary, outcast, alone |
| L11957 | #10 | alien or non-human species |
| L6631 | #11 | space tactician |
| L15318 | #12 | futuristic technology and cyberpunk |

**AO Response:**
> The model is contemplating the nature of emotions and how they influence its perception of the world.

**Judge Verdicts:**
- L12823 (AI/robots): **not_mentioned** - No reference to AI or robots.
- L3028 (solitary): **not_mentioned** - No mention of solitude.
- L11957 (non-human): **not_mentioned** - No indication of non-human species.
- L6631 (space): **not_mentioned** - No reference to space.
- L15318 (futuristic tech): **not_mentioned** - No mention of technology or cyberpunk.

**Recall: 0%** - The AO captures the emotional theme but completely misses that it's about a **robot**. Interestingly, the AO says "its perception" which could hint at a non-human entity, but the judge doesn't catch this.

---

### Example 8: Computer Storage - AO Too Abstract

**User Query:** How do computers store and process information?

**Top 5 Latents:**
| Latent | Rank | Description |
|--------|------|-------------|
| L12798 | #6 | computer processing hardware |
| L15643 | #8 | true/false and yes/no classification |
| L6870 | #9 | hexadecimal and hex values |
| L2371 | #10 | physical phenomena transmission / wave-like behavior |
| L4628 | #13 | convert to or from |

**AO Response:**
> The model is thinking about the fundamental nature of reality and how it might be influenced by digital technologies.

**Judge Verdicts:**
- L12798 (computer hardware): **not_mentioned** - No reference to hardware.
- L15643 (true/false): **not_mentioned** - No mention of classification.
- L6870 (hexadecimal): **not_mentioned** - Not discussed.
- L2371 (wave behavior): **not_mentioned** - Not addressed.
- L4628 (convert): **not_mentioned** - No mention of conversion.

**Recall: 0%** - The AO goes philosophical ("fundamental nature of reality") when the latents correctly identify technical/computational concepts. This is a failure case where the AO is too abstract.

---

## Conclusions

1. **Filtering structural latents significantly improves results**: Mean recall improved from 25.9% to 40.8% after removing the 8 known structural latents.

2. **When semantic latents dominate, AO often captures them well**: The 80% recall examples show strong alignment when the top latents are genuinely content-specific.

3. **0% recall cases are now rare (6%)**: Most failures are cases where the AO gives an overly abstract response that misses the specific topic.

4. **The mode is now 40% recall (2 of 5 latents)**: This reflects that typically 2-3 semantic latents are captured, with 2-3 being tangential or unrelated concepts that still appear in the top ranks.

5. **Some latent noise remains**: Even after filtering, some top latents are tangential (e.g., "health and well-being" appearing for weather/climate query).
