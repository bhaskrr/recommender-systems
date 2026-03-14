# Mean Average Precision (MAP)

## From AP to MAP - One Simple Step

Average Precision evaluates a ranked list for a **single user**. But a recommender system serves thousands or millions of users simultaneously. You need a single number that summarizes system-level performance across all of them.

That number is **Mean Average Precision (MAP)**, the arithmetic mean of AP scores across all users in your evaluation set:

$$MAP = \frac{1}{|U|} \sum_{u \in U} AP_u$$

Where:

- $|U|$ = total number of users in the evaluation set
- $AP_u$ = Average Precision for user $u$

That's genuinely all there is to MAP structurally. The complexity lives entirely inside AP, MAP is just its average. But the implications of that averaging are worth understanding carefully.

---

## Why Averaging AP is Non-Trivial

Averaging sounds simple, but consider what you're actually averaging over:

- Users with vastly different numbers of relevant items
- Users with sparse interaction histories vs dense ones
- Users whose preferences are well-represented in the training data vs cold-ish users

A user with 50 relevant items contributes one AP value to the mean. A user with 2 relevant items also contributes one AP value. The user with 2 relevant items has a noisier, more volatile AP (as noted in the previous file) - but MAP weights them equally.

This equal weighting is a deliberate choice. MAP treats every user as equally important regardless of how active they are. Whether that's the right choice depends on your system's goals:

- **Equal user weighting (MAP):** Every user's experience matters equally. Protects against optimizing only for power users.
- **Interaction-weighted averaging:** More active users get more weight. Biases toward users who generate more data.

MAP uses equal user weighting. Keep this in mind when interpreting results - a high MAP could be driven by excellent performance on users with many relevant items masking poor performance on sparse users.

---

## A Worked Example

Five users, each with their own AP score:

```bash
User 1:  AP = 0.80   (many relevant items, model performs well)
User 2:  AP = 0.45   (few relevant items, harder case)
User 3:  AP = 0.92   (model nails this user's preferences)
User 4:  AP = 0.30   (cold-ish user, model struggles)
User 5:  AP = 0.61   (average case)
```

$$MAP = \frac{0.80 + 0.45 + 0.92 + 0.30 + 0.61}{5} = \frac{3.08}{5} = 0.616$$

Now imagine a second model that improves User 4 significantly but slightly hurts User 3:

```bash
User 1:  AP = 0.80
User 2:  AP = 0.45
User 3:  AP = 0.85   (slightly worse)
User 4:  AP = 0.65   (much better)
User 5:  AP = 0.61

MAP = (0.80 + 0.45 + 0.85 + 0.65 + 0.61) / 5 = 3.36 / 5 = 0.672
```

Model 2 has higher MAP. It's better for the system overall even though it sacrificed some performance for User 3. MAP naturally captures this tradeoff, improvements for underserved users count exactly as much as improvements for well-served ones.

---

## MAP@K

Just as AP has an AP@K variant, MAP has **MAP@K**, the mean of AP@K scores across all users:

$$MAP@K = \frac{1}{|U|} \sum_{u \in U} AP@K_u$$

MAP@K is the standard form used in RecSys evaluation. K = 10 and K = 20 are most common. Always specify K when reporting MAP, "MAP" without a K value is ambiguous.

---

## MAP vs Other Aggregation Strategies

It's worth briefly understanding what MAP is not:

**MAP ≠ Precision@K averaged across users.** Mean Precision@K averages a position-blind metric. MAP averages a position-aware metric. They measure different things.

**MAP ≠ a single global AP computation.** You could pool all users' interactions into one giant ranked list and compute AP on that. This would weight users by their number of interactions, not equally. That's not MAP.

**MAP is a macro-average.** Each user gets one AP value regardless of how many items they've interacted with. This contrasts with micro-averaging, where each individual interaction gets equal weight.

---

## When MAP Is and Isn't the Right Metric

**MAP works well when:**

- Users have multiple relevant items (AP is stable and meaningful).
- You want to reward systems that consistently rank relevant items high across all users.
- You're comparing systems in an academic benchmark setting.

**MAP is less appropriate when:**

- You're using leave-one-out evaluation (one relevant item per user) - AP with a single relevant item is just the reciprocal rank, making MAP equivalent to MRR.
- Users have very different numbers of relevant items, the equal weighting may not reflect business priorities.
- You care about diversity or novelty, which MAP doesn't capture at all.

---

## MAP in Context - The RecSys Metric Landscape

At this point you have three metrics that build on each other:

```bash
Precision@K
    → position-blind within K
    → fixed by AP (rewards early relevant items)
        → per-user only
        → fixed by MAP (aggregates across all users)
            → still binary relevance only
            → fixed by NDCG (handles graded relevance)
```

MAP is the standard system-level metric for binary relevance evaluation. It's widely reported in IR and RecSys literature and gives you a single, principled number for comparing systems.

Its remaining limitation - binary relevance - is what NDCG addresses. In MovieLens, a 5-star rating is meaningfully more relevant than a 3-star rating. MAP treats both identically. NDCG doesn't.

But before NDCG, there's **MRR**, a simpler metric designed specifically for the case where each user has exactly one relevant item. That's the leave-one-out setting you'll use throughout this curriculum.

---

## Summary

| Aspect           | Detail                                                   |
| ---------------- | -------------------------------------------------------- |
| What it is       | Mean of AP scores across all users                       |
| Formula          | (1/\|U\|) Σ APᵤ                                          |
| Range            | [0, 1], higher is better                                 |
| User weighting   | Equal - every user contributes one AP value              |
| Limitation       | Binary relevance, noisy with few relevant items per user |
| Standard variant | MAP@10 or MAP@20 in RecSys papers                        |
