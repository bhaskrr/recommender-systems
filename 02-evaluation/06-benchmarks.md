# Benchmarks

A benchmark is more than just a dataset. It's a complete, reproducible evaluation protocol - a specific combination of:

```bash
Dataset + Preprocessing + Split strategy + Metrics + Baselines
```

Two papers can use "MovieLens 1M" and still produce incomparable results if they use different preprocessing, different splits, or different negative sampling strategies. A benchmark pins all of these decisions down so that results across papers are genuinely comparable.

This is why benchmarks matter: they're the shared language of progress in RecSys research. When a paper reports NDCG@10 on a standard benchmark, you can immediately compare it against dozens of prior results without worrying about experimental inconsistencies.

---

## Standard Benchmark Protocols

### The Leave-One-Out Protocol (Most Common in Neural RecSys)

Used in: NCF, SASRec, BERT4Rec, and most neural RecSys papers.

```bash
1. Filter:     Apply 5-core or 10-core filtering
2. Split:      For each user, hold out the last item as test,
               second-to-last as validation, rest as training
3. Negatives:  Sample 99 random negatives per test item
4. Evaluate:   Rank test item among 99 negatives
5. Metrics:    HR@10, NDCG@10, MRR@10
```

**Why 99 negatives?** Full catalog ranking for every user is expensive. Ranking 1 positive among 99 negatives is a practical approximation. The downside - as discussed in test collections - is that metrics computed this way are optimistic and not directly comparable to full-catalog evaluation.

**Temporal ordering matters:** The last item chronologically is held out, not a random item. This prevents data leakage - you don't want your model trained on interactions that happened after the test interaction.

---

### The Ratio Split Protocol (Common in CF and MF Papers)

Used in: matrix factorization papers, collaborative filtering baselines.

```bash
1. Filter:    Apply k-core filtering (typically 5-core or 10-core)
2. Split:     80% training / 10% validation / 10% test (random or temporal)
3. Evaluate:  Full catalog ranking or sampled negatives
4. Metrics:   NDCG@K, Recall@K, Precision@K
```

This protocol gives more test items per user than leave-one-out, making metrics more stable. The tradeoff is that random splitting can leak temporal information if interactions have a natural time ordering.

---

### The Strong Generalization Protocol (Used in Variational AutoEncoder Papers)

Used in: Mult-VAE, RecVAE, and related papers.

```bash
1. Split users (not interactions):
   - 80% of users → training users (all their interactions for training)
   - 10% of users → validation users
   - 10% of users → test users
2. For validation/test users:
   - 80% of their interactions → input (observed)
   - 20% of their interactions → held-out ground truth
3. Metrics:   NDCG@100, Recall@20, Recall@50
```

This protocol tests genuine generalization to unseen users — not just predicting held-out items for users seen during training. It's the hardest evaluation protocol and produces the most realistic measure of how a model performs on new users.

---

## Standard Baselines Every Paper Reports

When you implement a model, you need baselines to compare against. These are the ones that appear in virtually every RecSys paper:

### Non-Personalized Baselines

**MostPopular:** Recommend the globally most interacted-with items to every user. Deceptively strong - popularity bias means popular items are genuinely relevant to many users. Any personalized model should clearly outperform this.

**Random:** Recommend randomly sampled items. The floor baseline. If your model can't beat random, something is fundamentally wrong.

### Collaborative Filtering Baselines

**ItemKNN:** Item-based collaborative filtering with cosine similarity. Surprisingly competitive against neural models on small datasets. Fast and interpretable.

**UserKNN:** User-based collaborative filtering. Generally weaker than ItemKNN in practice.

**BPR-MF:** Matrix factorization trained with BPR loss. The standard MF baseline for implicit feedback datasets. You'll implement this in Module 3.

### Neural Baselines

**NCF (NeuMF):** Neural Collaborative Filtering. The standard neural baseline that most sequential and advanced models compare against. You'll implement this in Module 4.

**GRU4Rec:** RNN-based sequential model. The baseline for sequential recommendation - SASRec and BERT4Rec are evaluated against it.

---

## The Standard Benchmark Datasets

These are the exact dataset configurations that appear most frequently in papers. Using these exact settings makes your results directly comparable.

### MovieLens 1M - Standard CF/MF Benchmark

```bash
Source:     Grouplens
Filtering:  5-core (already satisfied in this dataset)
Split:      Leave-one-out (last rating as test, second-last as validation)
Negatives:  99 randomly sampled per test user
Metrics:    HR@10, NDCG@10
```

### Amazon Product Reviews - Standard Sequential Benchmark

```bash
Source:     https://nijianmo.github.io/amazon/
Subset:     Beauty, Sports and Outdoors, or Toys and Games (most commonly used)
Filtering:  5-core
Split:      Leave-one-out (chronological - last interaction as test)
Negatives:  99 randomly sampled per test user
Metrics:    HR@10, NDCG@10, MRR@10
```

### Yelp - Standard for Real-World Sparse Data

```bash
Source:     https://www.yelp.com/dataset
Filtering:  10-core
Split:      Leave-one-out (temporal)
Negatives:  99 randomly sampled per test user
Metrics:    HR@10, NDCG@10
```
