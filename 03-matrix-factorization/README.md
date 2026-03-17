# Module 03 - Matrix Factorization

## The Core Idea

Collaborative Filtering from Module 1 found similar users or items and used their known ratings to predict unknowns. It works, but it has a fundamental scalability problem, computing pairwise similarities across millions of users and items is expensive, and the sparse interaction matrix makes those similarities noisy.

Matrix Factorization takes a different approach entirely:

> **Decompose the user-item interaction matrix into two smaller matrices of latent factors. Represent every user and every item as a dense vector in a shared low-dimensional space. Predict interactions as dot products.**

This single idea, learned dense embeddings with dot product scoring, is the foundation of virtually every neural RecSys model you'll encounter in following modules. Understanding it thoroughly here makes everything downstream feel like a natural extension.

---

## Why This Module Comes Before Neural RecSys

Neural Collaborative Filtering (NCF), Two-Tower models, SASRec, BERT4Rec - all of them start with an embedding lookup. That embedding lookup is exactly the U and V matrices from Matrix Factorization, learned end-to-end. The architectural complexity in neural models is built on top of this foundation.

If you skip MF and jump straight to neural models, the embedding layers feel like magic. After this module, they'll feel obvious.

---

## Topics

### [01 SVD](./01-svd/)

The mathematical backbone of Matrix Factorization. How singular value decomposition factorizes a matrix into user and item latent factor matrices, why truncated SVD gives the best low-rank approximation, and how to adapt it for sparse interaction data by learning the factorization directly via gradient descent.

### [02 ALS](./02-als/)

Alternating Least Squares, a practical optimization alternative to SGD for Matrix Factorization. Particularly effective for implicit feedback datasets. Fixes one matrix and solves for the other in closed form, then alternates. Used in production systems at Spotify and Netflix.

### [03 NMF](./03-nmf/)

Non-negative Matrix Factorization, a variant that constrains all factor values to be non-negative. This constraint produces parts-based, interpretable representations. Less common in modern RecSys but important for understanding the space of MF variants.

### [04 BPR](./04-bpr/)

Bayesian Personalized Ranking, the most important topic in this module. Rather than minimizing rating prediction error (pointwise loss), BPR directly optimizes the ranking of relevant items above irrelevant ones (pairwise loss). This is the bridge between MF and production RecSys — virtually every implicit feedback model uses a variant of BPR loss.

---

## Implicit vs Explicit Feedback - The Critical Distinction

Before diving into implementations, understand this distinction - it affects every modeling and evaluation decision in this module.

**Explicit feedback:** The user directly expresses a preference. Star ratings, thumbs up/down, review scores. Easy to interpret but rare — most users never rate anything.

**Implicit feedback:** Behavior that implies preference without stating it. Clicks, purchases, watch time, streams, page views. Abundant but noisy — a click doesn't mean the user liked something, and no click doesn't mean they didn't.

```latex
Explicit:  r_ui ∈ {1, 2, 3, 4, 5}   — directly observed preference
Implicit:  r_ui ∈ {0, 1}             — 1 = interacted, 0 = unknown
                                        (not "disliked", just "not observed")
```

The key asymmetry in implicit data: **0 means unknown, not negative.** A user who never watched a movie might love it, they just haven't seen it. This breaks the standard regression objective (minimizing squared error on all entries) because you'd be penalizing the model for assigning high scores to items the user would actually like.

BPR is specifically designed to handle this, it only requires that observed items score higher than unobserved ones, making no assumption about the absolute value of unobserved scores.

---

## Dataset

All notebooks in this module use **MovieLens 100K**. Make sure it's downloaded before running any notebook:

Run the downloader notebook at `datasets/download_datasets.ipynb`.

---

## Concepts Introduced in This Module

| Concept | Where |
|---------|-------|
| Latent factor model | SVD |
| Truncated SVD | SVD |
| SGD for MF | SVD |
| User/item bias terms | SVD |
| Alternating Least Squares | ALS |
| Confidence-weighted implicit feedback | ALS |
| Non-negative constraints | NMF |
| Parts-based representation | NMF |
| Pairwise ranking loss | BPR |
| Negative sampling | BPR |
| Implicit feedback modeling | BPR |
