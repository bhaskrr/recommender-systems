# F-Measure

## The Problem F-Measure Solves

Precision and Recall are useful individually but awkward to work with together. If you're comparing two models:

```bash
Model A: Precision = 0.7, Recall = 0.4
Model B: Precision = 0.4, Recall = 0.7
```

Which is better? You can't tell without knowing which metric matters more for your use case. Comparing models across two numbers is also cumbersome when you're tuning hyperparameters or running experiments.

F-Measure solves this by combining Precision and Recall into a **single scalar** that captures both simultaneously.

---

## The Harmonic Mean — Why Not Just Average?

The naive approach would be to take the arithmetic mean:

$$\text{Arithmetic Mean} = \frac{P + R}{2}$$

But this has a critical flaw. Consider a degenerate model that recommends every single item in the catalog:

```bash
Precision = very low  (most recommendations are irrelevant)
Recall    = 1.0       (retrieved everything, so all relevant items are found)

Arithmetic mean = (0.01 + 1.0) / 2 = 0.505
```

That's a suspiciously high score for a completely useless model. The arithmetic mean gets fooled because one inflated value pulls the average up.

The **harmonic mean** fixes this. It penalizes extreme imbalance between the two values - both Precision and Recall need to be high to get a high score:

$$\text{Harmonic Mean} = \frac{2 \cdot P \cdot R}{P + R}$$

Same degenerate model:
$$F = \frac{2 \times 0.01 \times 1.0}{0.01 + 1.0} = \frac{0.02}{1.01} \approx 0.02$$

Much more honest. A model with precision = 0.01 scores near 0 regardless of its recall.

---

## F1 Score

The standard F-Measure is **F1** - it treats Precision and Recall as equally important:

$$F_1 = \frac{2 \cdot P \cdot R}{P + R}$$

This is equivalent to the harmonic mean of Precision and Recall.

**Properties:**

- Range: [0, 1]
- F1 = 1.0 only when both Precision = 1.0 and Recall = 1.0
- F1 = 0 if either Precision = 0 or Recall = 0
- F1 is closer to the lower of the two values - it gets dragged down by whichever metric is weaker

---

## A Worked Example

Using the same example from the previous file:

```bash
Recommended: [Movie A, Movie B, Movie C, Movie D, Movie E, Movie F]
Relevant in catalog: [Movie A, Movie C, Movie E, Movie G, Movie H]

Precision = 3/6 = 0.5
Recall    = 3/5 = 0.6
```

$$F_1 = \frac{2 \times 0.5 \times 0.6}{0.5 + 0.6} = \frac{0.6}{1.1} \approx 0.545$$

Now compare two hypothetical models:

```bash
Model A: P = 0.8, R = 0.2  →  F1 = (2 × 0.8 × 0.2) / (0.8 + 0.2) = 0.32 / 1.0 = 0.32
Model B: P = 0.5, R = 0.5  →  F1 = (2 × 0.5 × 0.5) / (0.5 + 0.5) = 0.50 / 1.0 = 0.50
```

Model B wins on F1 even though Model A has higher precision. F1 rewards balance - a model that does reasonably well on both metrics beats one that excels at one and neglects the other.

---

## The General Form

F1 assumes Precision and Recall are equally important. That's not always true. The general **Fβ** metric lets you control the balance:

$$F_\beta = \frac{(1 + \beta^2) \cdot P \cdot R}{\beta^2 \cdot P + R}$$

Where **β controls how much more you weight Recall over Precision:**

| β value | Weights                     | Use case                                        |
| ------- | --------------------------- | ----------------------------------------------- |
| β = 0.5 | Precision 2× more important | Short, curated recommendation lists             |
| β = 1.0 | Equal weight (F1)           | General purpose                                 |
| β = 2.0 | Recall 2× more important    | Candidate generation, don't miss relevant items |

**Intuition for β:**

- β < 1 --> you care more about not recommending bad items (precision-heavy)
- β > 1 --> you care more about not missing good items (recall-heavy)

---

## Verifying the Formula at the Extremes

It helps to sanity-check Fβ at the boundary cases:

**When β --> 0:** The formula collapses entirely to Precision. Recall disappears.

**When β --> ∞:** The formula collapses entirely to Recall. Precision disappears.

**When P = R:** Fβ = P = R regardless of β. When the two metrics are equal, the weighting doesn't matter.

---

## Why F-Measure Has Limited Use in RecSys

F-Measure is standard in classification and IR but sees **limited direct use in RecSys** for two reasons:

**1. It still ignores rank order.** Just like raw Precision and Recall, F-Measure treats a relevant item at position 1 identically to one at position 10. RecSys needs rank-aware metrics.

**2. Recall is often undefined.** As discussed in the previous file, you rarely know the complete set of relevant items for a user. With leave-one-out evaluation (one test item per user), computing meaningful recall requires special handling.

Where F-Measure does appear in RecSys is in **coverage and diversity evaluation** - measuring properties of the recommendation list beyond pure relevance - and occasionally in early-stage ablation studies where you want a single quick scalar to compare variants.

**Key takeaway:** F-Measure is a principled way to combine Precision and Recall into one number, but it inherits their shared blindspot - rank order.
