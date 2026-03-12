# Precision and Recall

## The Starting Point

Precision and Recall are the foundation of every retrieval metric we'll encounter in RecSys. They come from Information Retrieval (IR) - the field of finding relevant documents from a large collection - and were adapted for recommendation evaluation.

Before anything else, internalize this framing:

> A recommender system is essentially a **retrieval system**. Given a user, retrieve the items most relevant to them from a large catalog. Precision and Recall measure how well that retrieval works.

---

## The Setup

Imagine you have a user. The world is divided into two sets:

```bash
All items in the catalog
│
├── Items the user finds relevant     (ground truth positives)
└── Items the user finds irrelevant   (ground truth negatives)
```

Your model generates a recommendation list. That list is also divided:

```bash
Recommendation list
│
├── Items that are actually relevant  (true positives  — TP)
└── Items that are not relevant       (false positives — FP)
```

And among all relevant items, some were retrieved and some were missed:

```bash
All relevant items
│
├── Retrieved by the model            (true positives  — TP)
└── Missed by the model               (false negatives — FN)
```

---

## Precision

**Intuition:** Of everything you recommended, how much of it was actually useful?

$$\text{Precision} = \frac{\text{relevant items retrieved}}{\text{total items retrieved}} = \frac{TP}{TP + FP}$$

In plain terms: if you recommend 10 items and 4 of them are relevant, your precision is 0.4.

**What it punishes:** Recommending irrelevant items. Every bad recommendation hurts precision.

**What it ignores:** How many relevant items you missed. You could recommend just 1 item, make it perfectly relevant, and get precision = 1.0 — even if there were 50 other relevant items you didn't surface.

---

## Recall

**Intuition:** Of everything that was relevant, how much did you actually find?

$$\text{Recall} = \frac{\text{relevant items retrieved}}{\text{total relevant items}} = \frac{TP}{TP + FN}$$

In plain terms: if there are 20 relevant items for a user and your model retrieved 8 of them, recall is 0.4.

**What it punishes:** Missing relevant items. Every relevant item you fail to recommend hurts recall.

**What it ignores:** How many irrelevant items you recommended. You could recommend every single item in the catalog and get recall = 1.0 - because you'd retrieve all relevant items - even though your list is useless.

---

## The Precision-Recall Tradeoff

Precision and Recall pull in opposite directions. This tension is fundamental and unavoidable:

- **Recommend fewer items** → Precision goes up (you're more selective), Recall goes down (you miss more relevant items)
- **Recommend more items** → Recall goes up (you catch more relevant items), Precision goes down (more noise enters the list)

This tradeoff is why neither metric alone tells the full story. You always need both.

In practice, we choose an operating point based on what matters more for your use case:

- **High precision preferred:** The recommendation list is short and prominent (home screen hero slot). Every bad recommendation is costly.
- **High recall preferred:** You're doing candidate generation — surfacing 500 candidates from a million items for a downstream ranker to sort. Missing relevant items is more costly than including some noise.

---

## A Worked Example

Say a user has 5 relevant movies in the catalog. Your model recommends 6 movies:

```bash
Recommended list:
1. Movie A  relevant
2. Movie B  not relevant
3. Movie C  relevant
4. Movie D  not relevant
5. Movie E  relevant
6. Movie F  not relevant

Relevant items in catalog: [Movie A, Movie C, Movie E, Movie G, Movie H]
```

**Computing Precision:**
$$\text{Precision} = \frac{TP}{TP + FP} = \frac{3}{3 + 3} = \frac{3}{6} = 0.5$$

**Computing Recall:**
$$\text{Recall} = \frac{TP}{TP + FN} = \frac{3}{3 + 2} = \frac{3}{5} = 0.6$$

The model retrieved 3 out of 5 relevant items (recall = 0.6) but half its recommendations were noise (precision = 0.5).

---

## Precision and Recall in RecSys vs Classical IR

In classical IR (search engines), relevance judgments are collected by human annotators who explicitly label documents as relevant or not. In RecSys, you almost never have this luxury.

Instead, RecSys uses a proxy: **interaction = relevant, no interaction = assumed irrelevant**. This is the same relevance assumption introduced in `01-why-evaluation.md`. It's imperfect - a user may not have interacted with an item simply because they never saw it - but it's the only scalable option for offline evaluation.

This also means your ground truth set is whatever items the user interacted with in the test split. In a leave-one-out setup, each user has exactly **one** relevant test item. That has significant implications for how precision and recall behave.

---

## The Confusion Matrix View

For completeness, here's how the four outcomes map to a confusion matrix:

```bash
                    Predicted
                  Relevant | Not Relevant
                ┌──────────┬─────────────┐
Actual Relevant │    TP    │     FN      │
                ├──────────┼─────────────┤
Actual Not Rel. │    FP    │     TN      │
                └──────────┴─────────────┘

Precision = TP / (TP + FP)   ← row: predicted relevant
Recall    = TP / (TP + FN)   ← column: actually relevant
```

TN (true negatives — items correctly not recommended) are largely irrelevant in RecSys because the catalog is so large that almost everything is a true negative. Metrics that depend on TN (like accuracy) are therefore useless in this domain.

---

## Limitations in the RecSys Context

**Precision and Recall ignore rank order.** A system that puts all relevant items at the bottom of the list scores identically to one that puts them at the top - as long as the counts are the same. In RecSys, where users rarely scroll past the first few results, this is a serious flaw.

This limitation motivates everything in the next section - ranked retrieval metrics like Average Precision, NDCG, and MRR are all designed to fix exactly this problem.

**Recall is often incomputable in practice.** You'd need to know the complete set of relevant items for each user. In most datasets, you only have a small sample of interactions, so the true denominator (all relevant items) is unknown. This is why Recall@K (recall within the top K recommendations) is used instead of full recall.

---

## Summary

| Metric | Formula | Rewards | Blind to |
|--------|---------|---------|---------|
| Precision | TP / (TP + FP) | Recommending relevant items | Missing relevant items |
| Recall | TP / (TP + FN) | Finding all relevant items | Recommending irrelevant items |

Both metrics treat all positions in the recommendation list equally - which is their key limitation.
