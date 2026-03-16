# Normalized Discounted Cumulative Gain (NDCG)

## The Two Remaining Problems

By this point in the metric chain, we've fixed several limitations:

- Precision@K -> position-blind within K -> fixed by AP
- AP -> per-user only -> fixed by MAP
- MAP -> awkward for leave-one-out -> fixed by MRR

But every metric so far shares two remaining limitations:

**1. Binary relevance only.** An item is relevant (1) or not (0). A 5-star rating and a 3-star rating are treated identically. Real interaction data has graded relevance signals - ratings, dwell time, purchase vs click - that binary metrics throw away.

**2. Linear rank penalty.** MRR penalizes rank 2 vs rank 1 by 0.5 (1.0 -> 0.5). It penalizes rank 10 vs rank 9 by only 0.011 (0.111 -> 0.100). The penalty curve is a hyperbola. But user attention doesn't decay hyperbolically, it decays more like a logarithm. Users are very sensitive to positions 1-5 and increasingly indifferent beyond that.

**NDCG fixes both.** It supports graded relevance and applies a logarithmic position discount that more accurately models how user attention decays down a ranked list.

---

## Building NDCG from First Principles

NDCG is built in three steps. Understanding each step individually makes the full formula feel inevitable.

### Step 1: Gain

The **gain** at position $k$ is simply the relevance score of the item at that position:

$$\text{gain}(k) = \text{rel}(k)$$

In binary relevance: rel(k) ∈ {0, 1}.
In graded relevance: rel(k) ∈ {0, 1, 2, 3, 4, 5} (or whatever scale you use).

The gain formulation used in most RecSys papers is:

$$\text{gain}(k) = 2^{\text{rel}(k)} - 1$$

This exponential form amplifies the difference between high and low relevance scores. A 5-star item contributes gain = 31, a 3-star item contributes gain = 7, a 1-star item contributes gain = 1. This makes the metric more sensitive to truly excellent recommendations at the top.

For binary relevance: $2^1 - 1 = 1$ for relevant, $2^0 - 1 = 0$ for irrelevant. Same as before.

### Step 2: Discounted Cumulative Gain (DCG)

**Cumulative Gain** would just sum up all gains, but that ignores position entirely. The fix is to apply a **logarithmic discount** to each position:

$$DCG@K = \sum_{k=1}^{K} \frac{2^{\text{rel}(k)} - 1}{\log_2(k + 1)}$$

The discount $\frac{1}{\log_2(k+1)}$ decreases as position increases:

```bash
Position 1:  discount = 1/log₂(2) = 1/1.000 = 1.000
Position 2:  discount = 1/log₂(3) = 1/1.585 = 0.631
Position 3:  discount = 1/log₂(4) = 1/2.000 = 0.500
Position 5:  discount = 1/log₂(6) = 1/2.585 = 0.387
Position 10: discount = 1/log₂(11)= 1/3.459 = 0.289
```

A highly relevant item at position 1 contributes its full gain. The same item at position 10 contributes only 28.9% of its gain. This logarithmic decay is a deliberate modeling choice, it reflects that the marginal loss in user attention slows down as you go deeper into the list.

### Step 3: Normalization - From DCG to NDCG

DCG is not comparable across users. A user with 10 highly relevant items will have a much higher maximum possible DCG than a user with 2 relevant items. You can't average raw DCG values across users meaningfully.

The fix is to normalize by the **Ideal DCG (IDCG)** - the DCG you would get if you ranked all items in perfect relevance order:

$$IDCG@K = DCG@K \text{ of the perfect ranking}$$

$$NDCG@K = \frac{DCG@K}{IDCG@K}$$

Now NDCG is always in [0, 1]. NDCG = 1.0 means your ranking is perfect, the most relevant items are at the top. NDCG = 0 means no relevant items were retrieved.

---

## The Full Formula

$$NDCG@K = \frac{\sum_{k=1}^{K} \frac{2^{\text{rel}(k)} - 1}{\log_2(k+1)}}{\sum_{k=1}^{K} \frac{2^{\text{rel}^*_k} - 1}{\log_2(k+1)}}$$

Where $\text{rel}^*_k$ is the relevance of the $k$-th item in the **ideal** ranking (sorted by relevance descending).

---

## A Worked Example - Graded Relevance

A user has interacted with 5 movies with the following ratings:

```bash
Movie A: 5 stars  →  rel = 5
Movie C: 4 stars  →  rel = 4
Movie E: 3 stars  →  rel = 3
Movie G: 2 stars  →  rel = 2
Movie J: 1 star   →  rel = 1
```

Your model's ranked list (top 5):

```bash
Position 1: Movie C  rel=4
Position 2: Movie A  rel=5
Position 3: Movie X  rel=0  (irrelevant)
Position 4: Movie E  rel=3
Position 5: Movie G  rel=2
```

**Computing DCG@5:**

$$DCG@5 = \frac{2^4-1}{\log_2 2} + \frac{2^5-1}{\log_2 3} + \frac{2^0-1}{\log_2 4} + \frac{2^3-1}{\log_2 5} + \frac{2^2-1}{\log_2 6}$$

$$= \frac{15}{1.000} + \frac{31}{1.585} + \frac{0}{2.000} + \frac{7}{2.322} + \frac{3}{2.585}$$

$$= 15.000 + 19.560 + 0 + 3.014 + 1.161 = 38.735$$

**Computing IDCG@5 (perfect ranking: A, C, E, G, J → rel 5,4,3,2,1):**

$$IDCG@5 = \frac{31}{1.000} + \frac{15}{1.585} + \frac{7}{2.000} + \frac{3}{2.322} + \frac{1}{2.585}$$

$$= 31.000 + 9.464 + 3.500 + 1.292 + 0.387 = 45.643$$

**Computing NDCG@5:**

$$NDCG@5 = \frac{38.735}{45.643} = 0.849$$

The model scores 0.849 - close to perfect but penalized for placing Movie A (the 5-star item) at position 2 instead of position 1, and for including an irrelevant item at position 3.

---

## NDCG With Binary Relevance

In leave-one-out evaluation with one relevant test item, NDCG simplifies significantly:

- The only relevant item has rel = 1
- IDCG@K = 1.0 (one relevant item at position 1 gives maximum DCG)
- DCG@K = $\frac{1}{\log_2(\text{rank}+1)}$ if the item is in top K, else 0

$$NDCG@K = \frac{1}{\log_2(\text{rank}+1)} \quad \text{if rank} \leq K$$

This gives a smoother version of MRR, both reward early retrieval but NDCG uses a log discount while MRR uses a reciprocal. In practice they're highly correlated in leave-one-out settings but NDCG is considered more theoretically grounded.

```bash
Item at rank 1:  NDCG = 1/log₂(2) = 1.000   MRR = 1/1 = 1.000
Item at rank 2:  NDCG = 1/log₂(3) = 0.631   MRR = 1/2 = 0.500
Item at rank 3:  NDCG = 1/log₂(4) = 0.500   MRR = 1/3 = 0.333
Item at rank 5:  NDCG = 1/log₂(6) = 0.387   MRR = 1/5 = 0.200
Item at rank 10: NDCG = 1/log₂(11)= 0.289   MRR = 1/10= 0.100
```

NDCG penalizes lower ranks less harshly than MRR, its logarithmic decay is slower than MRR's hyperbolic decay. This means NDCG gives slightly more credit to a model that retrieves the item at rank 5 vs rank 10.

---

## Mean NDCG - Aggregating Across Users

Just like MAP averages AP, you average NDCG across all users:

$$\text{mean NDCG@K} = \frac{1}{|U|} \sum_{u \in U} NDCG@K_u$$

In papers this is simply reported as "NDCG@K" - the mean is implied.

---

## NDCG in the Full Metric Picture

Here's the complete metric chain you've now built:

```bash
Precision / Recall
    └── set-based, position-blind
         └── F-Measure: combines both, still position-blind
              └── Precision@K: truncates at K, still position-blind within K
                   └── AP: rewards early relevant items (binary)
                        └── MAP: AP averaged across users (binary)
                             └── MRR: MAP for one-relevant-item case
                                  └── NDCG: graded relevance + log discount ✓
```

NDCG sits at the top of this chain because it addresses every limitation below it:

- Rank-sensitive
- Graded relevance
- Normalized across users
- Logarithmic decay matches user attention

It is the dominant metric in modern RecSys papers. When you see a single metric reported for SASRec, BERT4Rec, or any neural RecSys model - it's almost always NDCG@10.

---

## Summary

| Component | Formula                | Purpose                                   |
| --------- | ---------------------- | ----------------------------------------- |
| Gain      | $2^{rel} - 1$          | Amplifies high relevance scores           |
| Discount  | $1/\log_2(k+1)$        | Penalizes lower positions logarithmically |
| DCG@K     | $\sum$ gain × discount | Raw rank-weighted relevance score         |
| IDCG@K    | DCG of perfect ranking | Normalization baseline                    |
| NDCG@K    | DCG / IDCG             | Normalized to [0,1]                       |

| Metric      | Relevance | Rank-sensitive | Normalized |
| ----------- | --------- | -------------- | ---------- |
| Precision@K | Binary    | ❌             | ✓          |
| AP / MAP    | Binary    | ✓              | ✓          |
| MRR         | Binary    | ✓              | ✓          |
| NDCG        | Graded    | ✓              | ✓          |
