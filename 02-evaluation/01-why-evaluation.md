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
