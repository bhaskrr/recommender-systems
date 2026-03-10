# Why Evaluation Matters in Recommender Systems

## The Core Problem

You've built a recommender system. It produces a list of items for every user. Now the obvious question is:

> **Is it any good?**

That question is harder than it sounds. Unlike classification where ground truth is clear (the label is right or wrong), recommendations live in a grayer space. If your system recommends a movie the user hasn't seen, you don't actually know if they'd like it — you only know what they've already interacted with. This fundamental uncertainty shapes everything about how we evaluate RecSys.

---

## Why Just Accuracy can not be used

Your instinct from general ML might be to use accuracy; what fraction of recommendations were correct? But this breaks down immediately in RecSys for two reasons:

1. **The sparsity problem**

    A typical user has interacted with less than 1% of available items. If your model recommends 10 items and the user has only rated 20 out of 10,000, even a perfect model can only "verify" a tiny fraction of its recommendations against known ground truth. Accuracy over a 99% sparse matrix is meaningless.

2. **Order matters**

    A recommendation is not just a set — it's a ranked list. Recommending the perfect item at position 10 is much worse than recommending it at position 1. Accuracy treats all positions equally and therefore misses the most important signal.

This is why RecSys has its own family of evaluation metrics, built specifically to handle sparsity and rank sensitivity.

---

## What Good Evaluation Actually Measures

A well-designed evaluation framework for RecSys needs to answer several distinct questions simultaneously:

**Relevance** — Are the recommended items actually useful to the user? This is the baseline. A recommendation that nobody wants is worthless regardless of how technically sophisticated the model is.

**Rank quality** — Are the most relevant items appearing at the top of the list? A system that buries the best recommendation at position 9 out of 10 is nearly as bad as not recommending it at all. Users rarely scroll past the first few results.

**Coverage** — Does the system recommend across a wide range of items, or does it only push popular items? A system that only ever recommends the top 100 most popular movies has poor coverage even if those recommendations are technically relevant.

**Serendipity** — Does the system ever surface something genuinely surprising and delightful? Pure relevance optimization often produces obvious recommendations. Serendipity measures the system's ability to introduce users to things outside their established taste profile.

**Novelty** — Are the recommendations things the user hasn't already seen or discovered on their own? Recommending *The Godfather* to a film enthusiast is not novel.

**Diversity** — Within a single recommendation list, are the items varied? Ten similar items in a single list is a poor experience even if each item is individually relevant.

Most evaluation frameworks focus heavily on relevance and rank quality because they're measurable offline. Coverage, serendipity, novelty, and diversity often require online evaluation (A/B tests) or carefully designed user studies.

---

## The Two Evaluation Paradigms

### Offline Evaluation

You hold out a portion of known user interactions as a test set, pretend the model hasn't seen them, generate recommendations, and measure how well the model recovers those held-out interactions.

**Advantages:** Fast, cheap, reproducible, no users required.

**Disadvantages:** Deeply limited by what you already know. You can only evaluate against items the user has already interacted with. Items the user would have loved but never encountered are invisible to offline evaluation — this is called **exposure bias**.

Offline evaluation is where NDCG, MRR, Precision@K, and most other metrics you'll learn live.

### Online Evaluation (A/B Testing)

You deploy two versions of the system to real users simultaneously and measure actual behavior — clicks, purchases, watch time, return visits.

**Advantages:** Measures real impact. Captures the full user experience including serendipity and novelty. Ground truth is actual behavior, not historical proxies.

**Disadvantages:** Expensive, slow, requires infrastructure, and introduces real risk (a bad recommendation system can genuinely harm user experience and business metrics).

In production systems, offline evaluation narrows the candidate models and online evaluation makes the final call.

---
