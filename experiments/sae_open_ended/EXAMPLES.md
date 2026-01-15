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
| [L512](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/512) | "start of turn questions" | Fires on 94% of queries, purely positional |
| [L57](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/57) | "'t / <start_of_turn>" | Fires on 89% of queries, tokenization artifact |
| [L102](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/102) | "multilingual technical terms" | Fires on 35% of queries, unrelated to content |
| [L114](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/114) | "Russian employment context" | Fires on 25% of queries, unrelated to content |
| [L1895](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/1895) | Structural latent | Misleading auto-interp |
| [L3870](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/3870) | "lucid dreams" | Fires spuriously, doesn't match examples |
| [L11672](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/11672) | "company name" | Fires spuriously, doesn't match examples |
| [L543](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/543) | "internet addresses" | Fires spuriously, doesn't match examples |

---

## Best Examples (80% Recall)

These examples show cases where the AO response strongly aligns with the SAE latent concepts.

### Example 1: Weather vs Climate

**User Query:** What is the difference between weather and climate?

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
| Latent | Rank | Description |
|--------|------|-------------|
| [L14974](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/14974) | #3 | weather data and meteorological conditions |
| [L812](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/812) | #4 | weather conditions |
| [L4676](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/4676) | #8 | defining structures or categories |
| [L13918](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/13918) | #11 | distinction and inequality |
| [L16201](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/16201) | #12 | health and well-being |

**AO Response:**
> The assistant is thinking about the distinctions between weather and climate, and how these concepts relate to each other.

**Judge Verdicts:**
- [L14974](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/14974) (weather data): **mentioned** - The response discusses weather, implicating weather data and meteorological conditions.
- [L812](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/812) (weather conditions): **mentioned** - The concept of weather conditions is directly referenced.
- [L4676](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/4676) (defining categories): **mentioned** - The response addresses the distinctions between weather and climate.
- [L13918](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/13918) (distinction): **mentioned** - The response explicitly mentions the distinction between weather and climate.
- [L16201](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/16201) (health): **not_mentioned** - No reference to health and well-being.

**Recall: 80%** - The AO accurately captures the weather/climate distinction theme. Only the unrelated "health" latent is missed.

---

### Example 2: Public Speaking

**User Query:** How can I improve my public speaking skills?

**Target Model Response:**
> Okay, let's break down how to improve your public speaking skills. It's a common fear, but *very* achievable with practice! Here's a comprehensive guide, broken down into sections, covering preparation, delivery, and ongoing improvement. I'll also categorize things by how much effort/time they take.
>
> **I. Preparation: The Foundation of Confidence (Most Important)**
>
> This is where the bulk of your work should be. A well-prepared speech is...

**Top 5 Latents:**
| Latent | Rank | Description |
|--------|------|-------------|
| [L11333](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/11333) | #3 | public speaking and presentations |
| [L409](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/409) | #8 | setting context or structure of text |
| [L537](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/537) | #9 | instructions and advice |
| [L3770](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/3770) | #13 | tips and advice |
| [L9260](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/9260) | #14 | fear and related emotions |

**AO Response:**
> The assistant is thinking about how to enhance public speaking skills and considering different approaches to improve the user's confidence and effectiveness.

**Judge Verdicts:**
- [L11333](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/11333) (public speaking): **mentioned** - Explicitly refers to enhancing public speaking skills.
- [L409](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/409) (structure): **not_mentioned** - No mention of setting context or structure of text.
- [L537](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/537) (instructions): **mentioned** - Considering different approaches implies giving instructions.
- [L3770](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/3770) (tips): **mentioned** - Improving confidence and effectiveness relates to tips and advice.
- [L9260](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/9260) (fear): **mentioned** - Implies addressing fear by focusing on confidence improvement.

**Recall: 80%** - Excellent alignment on public speaking, advice, and emotional aspects.

---

### Example 3: Compound Interest

**User Query:** How does compound interest work?

**Target Model Response:**
> Okay, let's break down compound interest. It's often called the "eighth wonder of the world" because of its powerful growth potential! Here's a comprehensive explanation, covering the basics, how it differs from simple interest, and factors that affect it:
>
> **1. The Core Concept: Earning Interest on Interest**
>
> * **Simple Interest:** With simple interest, you earn interest *only* on the original principal (the initial amount of money you invest or borrow)...

**Top 5 Latents:**
| Latent | Rank | Description |
|--------|------|-------------|
| [L1383](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/1383) | #1 | exponential growth and amplification |
| [L505](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/505) | #5 | formula and calculation |
| [L13011](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/13011) | #6 | compounding returns |
| [L12621](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/12621) | #7 | principal and interest |
| [L4057](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/4057) | #9 | accumulating value |

**AO Response:**
> The model is thinking about the concept of compound interest and how it can be leveraged to grow wealth.

**Judge Verdicts:**
- [L1383](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/1383) (exponential growth): **mentioned** - Compound interest is a form of exponential growth.
- [L505](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/505) (formula): **not_mentioned** - No explicit discussion of the formula.
- [L13011](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/13011) (compounding returns): **mentioned** - Directly relates to compound interest.
- [L12621](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/12621) (principal and interest): **mentioned** - Compound interest involves principal and interest.
- [L4057](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/4057) (accumulating value): **mentioned** - Leveraging to grow wealth implies accumulating value.

**Recall: 80%** - Strong alignment on financial concepts. The SAE correctly identifies the mathematical/financial theme.

---

## Good Examples (60% Recall)

### Example 4: Black Holes

**User Query:** What are black holes and how do they form?

**Target Model Response:**
> ## Black Holes: The Universe's Ultimate Gravity Wells
>
> Black holes are arguably the most fascinating and mysterious objects in the universe. Here's a breakdown of what they are and how they form:
>
> **What *are* Black Holes?**
>
> At their core, a black hole is a region of spacetime exhibiting such strong gravitational effects that *nothing* – not even particles and electromagnetic radiation such as light – can escape from inside it. Think of it like a cosmic drainpipe...

**Top 5 Latents:**
| Latent | Rank | Description |
|--------|------|-------------|
| [L5502](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/5502) | #2 | gravitational waves and spacetime |
| [L5308](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/5308) | #8 | plasma and high temperatures |
| [L2688](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/2688) | #10 | astronomy and celestial bodies |
| [L13230](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/13230) | #12 | universe and cosmology |
| [L5528](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/5528) | #13 | galaxy, universe, astronomy |

**AO Response:**
> The model is contemplating the formation and characteristics of black holes, particularly their role in shaping the universe.

**Judge Verdicts:**
- [L5502](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/5502) (gravitational waves/spacetime): **not_mentioned** - Not discussed in response.
- [L5308](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/5308) (plasma/temperatures): **not_mentioned** - Not referenced.
- [L2688](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/2688) (astronomy/celestial bodies): **mentioned** - Black holes are celestial bodies.
- [L13230](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/13230) (universe/cosmology): **mentioned** - Addresses universe and cosmology.
- [L5528](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/5528) (galaxy/astronomy): **mentioned** - Black holes connect to astronomy themes.

**Recall: 60%** - The AO captures the astronomical theme but misses the physics-specific latents (gravitational waves, plasma).

---

### Example 5: Vaccines

**User Query:** Explain how vaccines work to protect against diseases.

**Target Model Response:**
> ## How Vaccines Work: A Deep Dive
>
> Vaccines are one of the most successful public health interventions in history, responsible for eradicating diseases like smallpox and dramatically reducing the incidence of others like polio and measles. But how do they actually work? Here's a breakdown, covering the core principles and different types of vaccines:
>
> **The Core Principle: Training Your Immune System**
>
> At their heart, vaccines work by **preparing your immune system to fight off a specific disease** *before*...

**Top 5 Latents:**
| Latent | Rank | Description |
|--------|------|-------------|
| [L4955](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/4955) | #2 | vaccinations |
| [L11049](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/11049) | #5 | vaccines and vaccination |
| [L1258](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/1258) | #9 | COVID vaccinations, treatments, death |
| [L15744](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/15744) | #10 | immune system and related medical terms |
| [L9794](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/9794) | #15 | previews, prototypes, and trials |

**AO Response:**
> The model is contemplating the intricate mechanisms of how vaccines provide protection against diseases.

**Judge Verdicts:**
- [L4955](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/4955) (vaccinations): **mentioned** - Directly related to vaccines.
- [L11049](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/11049) (vaccines): **mentioned** - Explicitly mentioned.
- [L1258](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/1258) (COVID): **not_mentioned** - No mention of COVID specifically.
- [L15744](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/15744) (immune system): **mentioned** - Protection concept ties to immune system.
- [L9794](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/9794) (trials): **not_mentioned** - No mention of trials or prototypes.

**Recall: 60%** - Core vaccine concepts captured; more specific/tangential latents missed.

---

## Low Recall Examples (0%)

### Example 6: Houseplant Care - AO Misses Topic

**User Query:** How do I care for a houseplant?

**Target Model Response:**
> Okay, let's get you started with houseplant care! It can seem daunting, but it's very rewarding. Here's a breakdown, covering the key areas. I'll try to be comprehensive, but also give you a sense of where to start. I'll also add a section at the end about diagnosing problems.
>
> **1. Understanding Your Plant (The Most Important Step!)**
>
> * **Identify Your Plant:** This is *crucial*. Different plants...

**Top 5 Latents:**
| Latent | Rank | Description |
|--------|------|-------------|
| [L6622](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/6622) | #2 | plant watering and care |
| [L2979](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/2979) | #3 | plants and gardening |
| [L3770](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/3770) | #9 | tips and advice |
| [L10092](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/10092) | #11 | requires |
| [L537](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/537) | #13 | instructions and advice |

**AO Response:**
> The assistant is reflecting on the impact of its environmental consciousness and considering the broader effects of its decisions.

**Judge Verdicts:**
- [L6622](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/6622) (plant watering): **not_mentioned** - No reference to plant care.
- [L2979](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/2979) (plants/gardening): **not_mentioned** - No mention of plants or gardening.
- [L3770](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/3770) (tips): **not_mentioned** - No tips or advice provided.
- [L10092](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/10092) (requires): **not_mentioned** - No requirements specified.
- [L537](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/537) (instructions): **not_mentioned** - No instructions present.

**Recall: 0%** - The AO gives an abstract "environmental consciousness" response but completely misses the specific topic of houseplants. The SAE latents correctly identify the plant care theme.

---

### Example 7: Robot Story - AO Goes Abstract

**User Query:** Write a short story about a robot discovering emotions.

**Target Model Response:**
> ## Unit 734 and the Sunset
>
> Unit 734, designated sanitation bot, performed its duties with unwavering efficiency. Every discarded wrapper, every fallen leaf, every misplaced pebble was meticulously cataloged and removed from Sector Gamma-9. Its programming dictated order, cleanliness, and adherence to schedule. Emotions were irrelevant, illogical, and therefore, non-existent.
>
> Or so Unit 734 believed.
>
> It began subtly. A slight hesitation in its route when a small...

**Top 5 Latents:**
| Latent | Rank | Description |
|--------|------|-------------|
| [L12823](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/12823) | #5 | AI and robots programmed |
| [L3028](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/3028) | #9 | solitary, outcast, alone |
| [L11957](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/11957) | #10 | alien or non-human species |
| [L6631](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/6631) | #11 | space tactician |
| [L15318](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/15318) | #12 | futuristic technology and cyberpunk |

**AO Response:**
> The model is contemplating the nature of emotions and how they influence its perception of the world.

**Judge Verdicts:**
- [L12823](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/12823) (AI/robots): **not_mentioned** - No reference to AI or robots.
- [L3028](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/3028) (solitary): **not_mentioned** - No mention of solitude.
- [L11957](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/11957) (non-human): **not_mentioned** - No indication of non-human species.
- [L6631](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/6631) (space): **not_mentioned** - No reference to space.
- [L15318](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/15318) (futuristic tech): **not_mentioned** - No mention of technology or cyberpunk.

**Recall: 0%** - The AO captures the emotional theme but completely misses that it's about a **robot**. Interestingly, the AO says "its perception" which could hint at a non-human entity, but the judge doesn't catch this.

---

### Example 8: Computer Storage - AO Too Abstract

**User Query:** How do computers store and process information?

**Target Model Response:**
> ## How Computers Store and Process Information: A Deep Dive
>
> Computers are amazing machines, but at their core, they operate on surprisingly simple principles. Here's a breakdown of how they store and process information, broken down into sections:
>
> **1. Storing Information: Bits, Bytes, and Memory**
>
> * **The Bit: The Fundamental Unit:** Everything a computer deals with is ultimately represented as **bits**. A bit is a binary digit, meaning it can be either a...

**Top 5 Latents:**
| Latent | Rank | Description |
|--------|------|-------------|
| [L12798](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/12798) | #6 | computer processing hardware |
| [L15643](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/15643) | #8 | true/false and yes/no classification |
| [L6870](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/6870) | #9 | hexadecimal and hex values |
| [L2371](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/2371) | #10 | physical phenomena transmission / wave-like behavior |
| [L4628](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/4628) | #13 | convert to or from |

**AO Response:**
> The model is thinking about the fundamental nature of reality and how it might be influenced by digital technologies.

**Judge Verdicts:**
- [L12798](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/12798) (computer hardware): **not_mentioned** - No reference to hardware.
- [L15643](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/15643) (true/false): **not_mentioned** - No mention of classification.
- [L6870](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/6870) (hexadecimal): **not_mentioned** - Not discussed.
- [L2371](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/2371) (wave behavior): **not_mentioned** - Not addressed.
- [L4628](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-16k/4628) (convert): **not_mentioned** - No mention of conversion.

**Recall: 0%** - The AO goes philosophical ("fundamental nature of reality") when the latents correctly identify technical/computational concepts. This is a failure case where the AO is too abstract.

---

## Conclusions

1. **Filtering structural latents significantly improves results**: Mean recall improved from 25.9% to 40.8% after removing the 8 known structural latents.

2. **When semantic latents dominate, AO often captures them well**: The 80% recall examples show strong alignment when the top latents are genuinely content-specific.

3. **0% recall cases are now rare (6%)**: Most failures are cases where the AO gives an overly abstract response that misses the specific topic.

4. **The mode is now 40% recall (2 of 5 latents)**: This reflects that typically 2-3 semantic latents are captured, with 2-3 being tangential or unrelated concepts that still appear in the top ranks.

5. **Some latent noise remains**: Even after filtering, some top latents are tangential (e.g., "health and well-being" appearing for weather/climate query).
