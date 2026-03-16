# Beyond Relevance

## The Limitation of Relevance-Only Metrics

Every metric in the previous two sections - Precision@K, MAP, MRR, NDCG; measures one thing: how well the system retrieves items the user has already shown interest in.

That's a necessary condition for a good recommender. But it's not sufficient.

Consider a system that achieves NDCG@10 = 0.90 on MovieLens. Impressive number. But look closer:

- It recommends _Titanic_, _Forrest Gump_, and _The Shawshank Redemption_ to almost every user
- It never surfaces anything from the catalog's long tail
- Users who love obscure 1970s French cinema get the same list as everyone else
- The recommendations are technically "relevant" (users have rated similar films highly) but completely unsurprising

This system is optimizing a metric without delivering real value. Users don't return to a recommender because it tells them things they already know - they return because it occasionally surfaces something they wouldn't have found themselves.

Beyond-relevance metrics exist to measure these dimensions of recommendation quality that NDCG is blind to.

---

## Coverage

### Catalog Coverage

**Intuition:** What fraction of the available catalog does the system ever recommend?

$$\text{Catalog Coverage} = \frac{|\text{unique items recommended across all users}|}{|\text{total items in catalog}|}$$

A system with coverage = 0.02 recommends only 2% of the catalog, it's essentially a popularity filter. A system with coverage = 0.60 exposes users to 60% of available items.

**Why it matters:** Low coverage means the long tail of niche items is invisible to users. For platforms where content creators depend on discovery (Spotify for indie artists, Etsy for small sellers), low coverage directly harms the ecosystem.

**The coverage-relevance tradeoff:** Coverage and NDCG often pull in opposite directions. A system can trivially maximize coverage by recommending random items, but relevance collapses. The goal is healthy coverage without sacrificing relevance.

### User Coverage

A related metric: what fraction of users receive at least K recommendations?

$$\text{User Coverage} = \frac{|\text{users with} \geq K \text{ recommendations}|}{|\text{total users}|}$$

This matters for cold-start users. A system that can't generate recommendations for new users has low user coverage, and those users churn.

---

## Popularity Bias and Long-Tail Coverage

This deserves its own treatment because it's one of the most pervasive problems in production RecSys.

**The Matthew effect:** In any interaction dataset, a small fraction of items (the "head") accumulates the vast majority of interactions. Models trained on this data learn that popular items are "safe" recommendations, they have abundant training signal and rarely produce bad predictions.

The result is a feedback loop:

```bash
Popular items get recommended
    → Popular items get more interactions
        → Popular items get even stronger training signal
            → Popular items get recommended even more
```

This loop systematically disadvantages niche items regardless of how well they'd actually match a particular user's taste.

**Measuring long-tail coverage:**

$$\text{Long-Tail Coverage} = \frac{|\text{long-tail items recommended}|}{|\text{total long-tail items}|}$$

Where "long-tail items" are typically defined as items below a popularity threshold (e.g., items outside the top 20% by interaction count).

**Average Popularity of Recommendations:**

$$\text{ARP} = \frac{1}{|U|} \sum_{u \in U} \frac{1}{K} \sum_{i \in L_u} \phi(i)$$

Where $\phi(i)$ is the popularity of item $i$ (interaction count) and $L_u$ is user $u$'s recommendation list. Lower ARP indicates the system recommends less popular, more niche items.

---

## Diversity

**Intuition:** Within a single recommendation list, how different are the items from each other?

A list of ten near-identical items is a poor experience even if each item is individually relevant. If you watch one action movie, getting nine more action movies is overwhelming, you'd want some variety.

### Intra-List Diversity (ILD)

The standard diversity metric is **Intra-List Diversity**, the average pairwise dissimilarity between items in a recommendation list:

$$ILD(L_u) = \frac{1}{K(K-1)} \sum_{i \in L_u} \sum_{j \in L_u, j \neq i} \text{dist}(i, j)$$

Where $\text{dist}(i, j)$ is a dissimilarity measure between items, typically 1 minus cosine similarity of their feature vectors (genre vectors, embeddings, etc.).

ILD = 0 means all items are identical. ILD = 1 means all items are maximally dissimilar.

**The diversity-relevance tradeoff:** Maximizing ILD by recommending maximally dissimilar items destroys relevance. The practical goal is a diverse list where items are still individually relevant, the user gets variety without getting random noise.

---

## Serendipity

**Intuition:** Does the system ever recommend something surprising that the user ends up loving?

Serendipity is the hardest beyond-relevance property to measure because it requires capturing both surprise and relevance simultaneously:

$$\text{Serendipity}(L_u) = \frac{1}{K} \sum_{i \in L_u} \text{unexpected}(i, u) \times \text{rel}(i, u)$$

Where $\text{unexpected}(i, u)$ measures how surprising item $i$ is for user $u$ , typically operationalized as how different $i$ is from the user's historical taste profile.

**The core tension:** An item can be surprising but irrelevant (random noise), or relevant but unsurprising (obvious recommendation). Serendipity requires both. A truly serendipitous recommendation is one the user wouldn't have found themselves, but loves once they see it.

**In practice:** Serendipity is notoriously hard to measure offline. The most reliable approach is online evaluation, measuring whether users who receive serendipitous recommendations show higher long-term engagement than those who receive only obvious ones.

---

## Novelty

**Intuition:** Are the recommendations items the user hasn't already discovered on their own?

Novelty is distinct from serendipity, it's purely about whether the item is new to the user, regardless of surprise:

$$\text{Novelty}(L_u) = \frac{1}{K} \sum_{i \in L_u} -\log_2 \phi(i)$$

Where $\phi(i)$ is the normalized popularity of item $i$. This is essentially the self-information of each recommendation, rare items carry more information (novelty) than common ones.

Lower popularity -> higher novelty. Recommending _The Godfather_ to a cinephile has near-zero novelty. Recommending an obscure 1973 Italian giallo they've never heard of has high novelty.

**Relationship to popularity bias:** Novelty and long-tail coverage are closely related. A system that overcomes popularity bias tends to recommend novel items as a side effect.

---

## Fairness

A dimension that's increasingly important in production systems but often absent from academic benchmarks.

**Provider fairness:** Are items from different providers (artists, sellers, content creators) recommended at rates proportional to their quality, not just their existing popularity? A streaming platform that never recommends indie artists regardless of their quality is unfair to those providers.

**User fairness:** Does the system perform equally well across different user groups (demographic groups, new vs. returning users, power users vs. casual users)? A system with high average NDCG that works well for power users but poorly for casual users is unfair.

**Measuring fairness:**

$$\Delta_{NDCG} = |NDCG_{\text{group A}} - NDCG_{\text{group B}}|$$

A large gap indicates the system serves one group significantly better than another.

---

## Putting It Together — A Balanced Evaluation Framework

In practice, you need a balanced set of metrics that captures multiple dimensions:

```bash
Relevance quality     →  NDCG@10, MRR@10, Hit Rate@10
Catalog health        →  Coverage, Long-Tail Coverage, ARP
List quality          →  Intra-List Diversity
Discovery quality     →  Novelty, Serendipity (online preferred)
Fairness              →  Per-group NDCG gap
```

No single metric captures everything. The right combination depends on your system's goals:

| System type            | Primary metrics | Secondary metrics           |
| ---------------------- | --------------- | --------------------------- |
| E-commerce             | NDCG, Hit Rate  | Coverage, Fairness          |
| Music streaming        | NDCG, Diversity | Novelty, Long-Tail Coverage |
| News feed              | MRR, Hit Rate   | Diversity, Serendipity      |
| Sequential (next-item) | NDCG@10, MRR@10 | Hit Rate@10                 |

## Summary

| Metric             | Measures                               | Offline? |
| ------------------ | -------------------------------------- | -------- |
| Catalog Coverage   | Fraction of catalog recommended        | ✓        |
| Long-Tail Coverage | Fraction of niche items surfaced       | ✓        |
| ARP                | Average popularity of recommendations  | ✓        |
| ILD                | Diversity within a recommendation list | ✓        |
| Novelty            | Self-information of recommendations    | ✓        |
| Serendipity        | Surprising + relevant recommendations  | Partial  |
| Fairness (ΔNDCG)   | Performance gap across user groups     | ✓        |
