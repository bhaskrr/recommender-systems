# Test Collections in Recommender Systems

## What is a Test Collection?

Before you can evaluate a recommender system, you need something to evaluate it against. In RecSys, that something is called a **test collection** — a standardized dataset of user-item interactions that the research community has agreed to use as a common benchmark.

A test collection has three components:

```
Test Collection
├── A corpus of items         (movies, products, songs, etc.)
├── A set of users            (with demographic info sometimes)
└── Relevance judgments       (ratings, clicks, purchases, implicit signals)
```

The quality of your evaluation is only as good as the quality of your test collection. A poorly constructed dataset produces misleading metrics, which produce misleading conclusions about which model is "better."

---

## The Standard Benchmarks

### MovieLens (GroupLens Research)

The most widely used RecSys benchmark by a large margin. Collected from the MovieLens movie rating website.

| Version | Users | Movies | Ratings | Best used for |
|---------|-------|--------|---------|---------------|
| ML-100K | 943 | 1,682 | 100,000 | Learning, prototyping |
| ML-1M | 6,040 | 3,706 | 1,000,209 | Standard benchmarking |
| ML-10M | 71,567 | 10,681 | 10,000,054 | Scalability testing |
| ML-20M | 138,493 | 27,278 | 20,000,263 | Large-scale experiments |

**Format:** Explicit ratings on a 1–5 scale with timestamps.

**Why it dominates:** Clean, well-documented, stable across versions, and used in virtually every RecSys paper since the Netflix Prize era. When you publish results on MovieLens, they're immediately comparable to a decade of prior work.

**Limitations:**

- Explicit ratings only - real-world systems mostly deal with implicit feedback
- Ratings are sparse but not as sparse as real production data
- The user population is self-selected (people who voluntarily rate movies)
- No cold-start users - everyone has a minimum number of ratings

---

### Amazon Product Reviews

A large-scale dataset of product reviews scraped from Amazon, covering multiple product categories.

**Key characteristics:**

- Available per category: Books, Electronics, Movies & TV, Clothing, etc.
- Contains both explicit ratings (1–5 stars) and review text
- Naturally sparse - most users review very few products
- Has temporal information (useful for sequential models)

**Best used for:** Sequential recommendation (SASRec, BERT4Rec), cross-domain recommendation, NLP-augmented RecSys (since you have review text).

**Limitations:**

- Noisy - many reviews are not genuine preference signals (gift purchases, returns, etc.)
- Heavy long-tail distribution - a small number of products have most of the reviews

---

### LastFM (HetRec 2011)

A dataset of music listening events from Last.fm, containing implicit feedback — play counts rather than explicit ratings.

**Key characteristics:**

- 1,892 users, 17,632 artists
- 92,834 user-artist listening records (play counts)
- No explicit ratings — pure implicit feedback
- Social network data available (friend connections)

**Best used for:** Implicit feedback models, social recommendation experiments.

**Why it matters:** LastFM forces you to work with implicit signals, which is much closer to how real systems operate. There are no ratings — you only know that a user listened to an artist, and how many times.

---

### Netflix Prize Dataset

The dataset from the famous Netflix Prize competition (2006–2009), which offered $1M for a 10% improvement over Netflix's existing algorithm.

**Key characteristics:**
- 480,189 users, 17,770 movies
- 100,480,507 ratings (explicit, 1–5 scale)
- Timestamps included

**Important caveat:** This dataset is no longer publicly available due to a privacy lawsuit. It established many RecSys conventions but you cannot use it in new work. It's worth knowing about for historical context — most of modern matrix factorization was developed on this dataset.

---

## How Test Collections Are Constructed

Understanding how a dataset was built tells you what it can and cannot measure.

### Step 1: Interaction Logging

Raw interactions are logged from a platform - ratings, clicks, purchases, streams, dwell time. The choice of interaction type determines whether you get explicit or implicit feedback.

### Step 2: Filtering (Pruning)

Raw logs are extremely noisy and sparse. Standard practice is to apply **k-core filtering** — keep only users with at least k interactions and items with at least k interactions. MovieLens 100K uses a 20-core filter (every user has rated at least 20 movies).

This is important to understand because **k-core filtering removes cold-start users and items entirely**. Your evaluation metrics on a k-core filtered dataset are optimistic — they don't reflect model performance on the hardest cases.

### Step 3: Train/Test Splitting

Several strategies exist:

**Random split** — randomly assign interactions to train/test. Simple but breaks temporal ordering. Not recommended for sequential models.

**Temporal split** — interactions before time T go to train, after T go to test. Respects the natural chronology of user behavior. More realistic but harder to implement consistently.

**Leave-one-out (LOO)** — for each user, hold out their most recent interaction as the test item, use everything else for training. Very common in academic papers because it gives one clean test item per user.

```bash
User interaction history (chronological):
[Item A, Item B, Item C, Item D, Item E] , Item E is the test item (held out)
[Item A, Item B, Item C, Item D] are used for training.
```

**Leave-last-k-out** - hold out the last **k** interactions per user. Used when you want to evaluate on multiple test items per user.

### Step 4: Negative Sampling

Here's something non-obvious: when you evaluate top-K recommendations, you don't actually rank all items for every user - that would be computationally prohibitive at scale. Instead, most academic evaluations use **negative sampling**:

For each user's test item, sample 99 (or some fixed number) of random items the user hasn't interacted with. Rank the test item among those 99 negatives. If the test item appears in the top K, it's a hit.

This is a significant approximation. It means your evaluation metrics are computed over 100 items, not the full catalog of thousands. Models evaluated this way look better than they would under full-catalog evaluation - keep this in mind when comparing results across papers.

---

## Dataset Biases You Must Know

Every test collection carries biases that affect what your evaluation metrics actually measure. Ignoring these leads to overly optimistic conclusions.

### Popularity Bias

In any interaction dataset, a small fraction of items accounts for the vast majority of interactions. In MovieLens, the top 20% of movies have roughly 80% of all ratings. This means:

- Models that recommend popular items get rewarded by offline metrics even if their recommendations aren't genuinely personalized
- The long tail of niche items is systematically underrepresented in training and evaluation
- A model that only recommends blockbusters can achieve surprisingly good Precision@K

**Consequence:** High offline metrics don't guarantee good tail coverage or genuine personalization.

### Exposure Bias

Users can only rate items they've been exposed to. If a platform's existing recommendation system is biased toward certain items, the logged interactions will reflect that bias - not true user preferences.

**Consequence:** Your model learns from and is evaluated against a biased sample of possible preferences. Items that were never surfaced to users are invisible to evaluation, even if users would have loved them.

### Selection Bias (Explicit Feedback)

Users who explicitly rate items are not representative of all users. People who take the time to rate movies tend to be more engaged, have stronger opinions, and interact with more items than the average user.

**Consequence:** Models trained on explicit ratings may not generalize to the broader user population.

### Survivorship Bias

Datasets often only contain interactions that "succeeded" in some sense - the user watched the movie to completion, left a review, made a purchase. Abandoned sessions, skipped items, and negative implicit signals are often not logged.

---

## A Note on Reproducibility

A persistent problem in RecSys research is that reported results are often not reproducible - different papers use the same dataset name but different preprocessing, different train/test splits, and different negative sampling strategies. This makes direct comparison of numbers across papers unreliable.

When you implement models in this repo, always document:

- Which dataset version you used
- What filtering was applied (k-core value)
- How you split train/test (random, temporal, LOO)
- How many negatives were sampled for evaluation
- What K values you used for top-K metrics

This discipline makes your results reproducible and your repo genuinely useful to others.

---

## Summary

| Concept | Key point |
|---|---|
| Test collection | Corpus + users + relevance judgments |
| k-core filtering | Removes cold-start cases — metrics are optimistic |
| Leave-one-out split | Most common academic protocol |
| Negative sampling | Evaluation over 100 items, not full catalog |
| Popularity bias | Popular items inflate offline metrics |
| Exposure bias | You only evaluate what users were shown |