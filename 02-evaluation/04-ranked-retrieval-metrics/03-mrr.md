# Mean Reciprocal Rank (MRR)

## The Setup - One Relevant Item Per User

MAP and AP are designed for users with multiple relevant items. But in leave-one-out evaluation, the most common protocol in RecSys papers, each user has exactly **one** relevant test item. In this setting, AP reduces to a single precision value at one position, and MAP becomes an average of those single values.

There's a cleaner, more intuitive metric built specifically for this case: **Mean Reciprocal Rank (MRR)**.

MRR asks a single, direct question:

> **Where does the one relevant item appear in the ranked list?**

The earlier it appears, the better. MRR quantifies this with a reciprocal, rank 1 is perfect, rank 2 is half as good, rank 10 is one-tenth as good, and so on.

---

## Reciprocal Rank - Per User

For a single user, the **Reciprocal Rank (RR)** is:

$$RR_u = \frac{1}{\text{rank of the first relevant item}}$$

Where "rank of the first relevant item" is its position in the ranked recommendation list (1-indexed).

```bash
Test item at position 1  →  RR = 1/1  = 1.000
Test item at position 2  →  RR = 1/2  = 0.500
Test item at position 3  →  RR = 1/3  = 0.333
Test item at position 5  →  RR = 1/5  = 0.200
Test item at position 10 →  RR = 1/10 = 0.100
Test item not in list    →  RR = 0
```

The reciprocal function creates a natural diminishing return - the penalty for being at rank 2 vs rank 1 is large (1.0 -> 0.5), but the penalty for being at rank 9 vs rank 10 is small (0.111 -> 0.100). This reflects real user behavior: users are much more sensitive to the difference between position 1 and 2 than between position 9 and 10.

---

## Mean Reciprocal Rank - Across All Users

**MRR** is simply the mean of Reciprocal Rank values across all users:

$$MRR = \frac{1}{|U|} \sum_{u \in U} \frac{1}{\text{rank}_u}$$

Where $\text{rank}_u$ is the position of the relevant test item for user $u$.

---

## A Worked Example

Six users, leave-one-out evaluation. Each user has one test item:

```bash
User 1: test item found at rank 1  →  RR = 1/1  = 1.000
User 2: test item found at rank 3  →  RR = 1/3  = 0.333
User 3: test item found at rank 2  →  RR = 1/2  = 0.500
User 4: test item found at rank 8  →  RR = 1/8  = 0.125
User 5: test item not in top 10    →  RR = 0.000
User 6: test item found at rank 1  →  RR = 1/1  = 1.000
```

$$MRR = \frac{1.000 + 0.333 + 0.500 + 0.125 + 0.000 + 1.000}{6} = \frac{2.958}{6} = 0.493$$

---

## MRR vs MAP in Leave-One-Out Setting

When each user has exactly one relevant item, AP@K for that user equals:

$$AP@K_u = \frac{1/\text{rank}_u}{1} = \frac{1}{\text{rank}_u} = RR_u \quad \text{(if rank}_u \leq K\text{)}$$

So MAP@K and MRR@K are **identical** in the leave-one-out setting. This is the equivalence mentioned at the end of the MAP file. In practice, papers using leave-one-out evaluation report MRR rather than MAP because the name more clearly communicates what's being measured.

---

## MRR@K - Truncating the Rank

In practice you don't evaluate over the full catalog - you evaluate within a fixed cutoff K. **MRR@K** sets RR to 0 if the relevant item doesn't appear in the top K:

$$RR@K_u = \begin{cases} \frac{1}{\text{rank}_u} & \text{if rank}_u \leq K \\ 0 & \text{otherwise} \end{cases}$$

$$MRR@K = \frac{1}{|U|} \sum_{u \in U} RR@K_u$$

MRR@10 is the standard variant in RecSys papers. Items ranked beyond position 10 are treated as not found.

---

## What MRR Captures and What It Misses

**What MRR captures well:**

- The position of the single most relevant item
- Whether the system is reliably surfacing the right item near the top
- Performance in question-answering style tasks where there's one right answer

**What MRR misses:**

- Performance beyond the first relevant item - if a user has multiple relevant items (even just 2), MRR ignores everything after the first hit
- The quality of the rest of the list - a system that gets rank 1 right but fills positions 2-10 with garbage scores identically to one with a high-quality full list
- Graded relevance - a 5-star item at rank 3 scores the same as a 2-star item at rank 3

**Consequence:** MRR is appropriate when the task is genuinely about finding one specific item (next-item prediction in sequential RecSys, for example). For tasks where the whole list quality matters, NDCG is more appropriate.

---

## MRR in the Context of Sequential RecSys

MRR has particular relevance to **sequential recommendation**. In sequential RecSys, you observe a user's interaction history and predict their next interaction. That's exactly a one-relevant-item task:

```bash
History:  [Item A -> Item B -> Item C -> Item D]
Question: What does the user interact with next?
Answer:   Item E  (one correct answer)
```

---

## The Metric Trio for Leave-One-Out Evaluation

In practice, leave-one-out evaluation in RecSys papers always reports these three metrics together - they're complementary:

```bash
Hit Rate@K   ->  Did we retrieve the item at all? (binary)
MRR@K        ->  How high did we rank it?         (rank-sensitive)
NDCG@K       ->  How high, with position weighting?
```

HR@K tells you coverage. MRR@K tells you rank quality. NDCG@K refines rank quality further with a logarithmic discount that more closely models user attention decay.

---

## Summary

| Aspect           | Detail                                             |
| ---------------- | -------------------------------------------------- |
| What it measures | Position of the first (only) relevant item         |
| Formula          | mean of 1/rank across all users                    |
| Range            | [0, 1], higher is better                           |
| Best used for    | Leave-one-out evaluation, next-item prediction     |
| Limitation       | Ignores items after first hit, no graded relevance |
| Standard variant | MRR@10 in RecSys papers                            |
