# SVD - Singular Value Decomposition for Recommender Systems

## Intuition

Every user has tastes. Every item has characteristics. The hypothesis behind Matrix Factorization is that both tastes and characteristics can be described by a small number of **latent factors** — hidden dimensions that explain why certain users gravitate toward certain items.

You don't define these factors manually. The model discovers them from the data. On MovieLens, they might loosely correspond to things like "cerebral vs action-packed" or "mainstream vs arthouse" — but they're not interpretable by design. They're whatever dimensions best explain the observed rating patterns.

Once you have these factors, everything simplifies:

```bash
User A's factors:  [0.9, 0.2, 0.7]   ← strong cerebral, weak action, strong arthouse
Item  X's factors: [0.8, 0.1, 0.6]   ← mostly cerebral, low action, arthouse

Predicted rating = dot product = 0.9×0.8 + 0.2×0.1 + 0.7×0.6 = 0.72 + 0.02 + 0.42 = 1.16
```

Scale this up and normalize it to the rating range, and you have a recommendation engine.

---

## The Math

### Classical SVD

Any real matrix R of shape (m × n) can be decomposed as:

$$R = U \Sigma V^T$$

- **U** — (m × m) orthogonal matrix. Each row is a user represented in factor space.
- **Σ** — (m × n) diagonal matrix. Diagonal entries σ₁ ≥ σ₂ ≥ ... ≥ 0 are singular values — they measure how much variance each factor explains.
- **Vᵀ** — (n × n) orthogonal matrix. Each column is an item represented in factor space.

### Truncated SVD

Keep only the top k singular values and their corresponding vectors:

$$R \approx U_k \Sigma_k V_k^T$$

This is the **best rank-k approximation** of R in terms of Frobenius norm — no other rank-k matrix is closer to R. The approximation error is:

$$||R - U_k \Sigma_k V_k^T||_F^2 = \sum_{i=k+1}^{\min(m,n)} \sigma_i^2$$

The discarded singular values tell you how much information you're losing. If σ₁ ≫ σ₂ ≫ ... ≫ σ*k ≫ σ*{k+1}, the truncation is clean. If the singular values decay slowly, you need a larger k.

The user embedding matrix is $P = U_k \Sigma_k^{1/2}$ and the item embedding matrix is $Q = V_k \Sigma_k^{1/2}$, so that $R \approx PQ^T$.

### The Sparsity Problem

Classical SVD requires a complete matrix. The MovieLens interaction matrix is ~94% sparse. Two approaches:

**Mean imputation:** Fill missing values with the global mean rating, run SVD. Simple but noisy — treats absence of rating as a neutral rating.

**Direct optimization (preferred):** Learn P and Q by minimizing prediction error only on observed entries:

$$\mathcal{L} = \sum_{(u,i) \in \mathcal{O}} (r_{ui} - \mathbf{p}_u \cdot \mathbf{q}_i)^2 + \lambda(||\mathbf{p}_u||^2 + ||\mathbf{q}_i||^2)$$

This is optimized via SGD — sample an observed rating, compute the error, backpropagate through the dot product, update $\mathbf{p}_u$ and $\mathbf{q}_i$.

### Adding Bias Terms

Raw MF predicts $\hat{r}_{ui} = \mathbf{p}_u \cdot \mathbf{q}_i$. But ratings are influenced by factors beyond user-item affinity:

- Some users rate everything highly (user bias $b_u$)
- Some items are universally loved or hated (item bias $b_i$)
- The global average rating pulls everything toward the center ($\mu$)

The full model with biases:

$$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u \cdot \mathbf{q}_i$$

The objective becomes:

$$\mathcal{L} = \sum_{(u,i) \in \mathcal{O}} (r_{ui} - \mu - b_u - b_i - \mathbf{p}_u \cdot \mathbf{q}_i)^2 + \lambda(||\mathbf{p}_u||^2 + ||\mathbf{q}_i||^2 + b_u^2 + b_i^2)$$

Bias terms typically improve RMSE by 0.05–0.10 on MovieLens — a significant gain for a small addition.

---

## SGD Update Rules

For each observed rating $r_{ui}$, compute the error:

$$e_{ui} = r_{ui} - \hat{r}_{ui}$$

Update rules (gradient descent on $\mathcal{L}$):

$$b_u \leftarrow b_u + \alpha (e_{ui} - \lambda b_u)$$
$$b_i \leftarrow b_i + \alpha (e_{ui} - \lambda b_i)$$
$$\mathbf{p}_u \leftarrow \mathbf{p}_u + \alpha (e_{ui} \mathbf{q}_i - \lambda \mathbf{p}_u)$$
$$\mathbf{q}_i \leftarrow \mathbf{q}_i + \alpha (e_{ui} \mathbf{p}_u - \lambda \mathbf{q}_i)$$

Where $\alpha$ is the learning rate.

---

## Hyperparameters

| Hyperparameter     | Typical range | Effect                                               |
| ------------------ | ------------- | ---------------------------------------------------- |
| k (latent factors) | 10–200        | Higher k → more expressive, slower, risk overfitting |
| λ (regularization) | 0.001–0.1     | Higher λ → stronger regularization, less overfitting |
| α (learning rate)  | 0.001–0.01    | Higher α → faster convergence, risk instability      |
| Epochs             | 20–100        | More epochs → better fit on train, monitor val loss  |

Start with k=50, λ=0.01, α=0.005 on MovieLens 100K — these are stable defaults.

---

## Evaluation Protocol

Leave-one-out on MovieLens 100K:

```
For each user:
  - Training:   all ratings except the last
  - Validation: second-to-last rating
  - Test:       last rating (chronologically)

Negative sampling: 99 random unrated items per test user
Metrics: HR@10, MRR@10, NDCG@10
```

---

## What's in the Notebook

`svd.ipynb` walks through:

1. **Data loading and preprocessing** — load MovieLens 100K, build user/item index maps, chronological train/val/test split
2. **Truncated SVD baseline** — impute missing values, run `torch.linalg.svd`, evaluate
3. **MF with SGD** — implement the full MF model in PyTorch with embedding layers, train with SGD, plot training curve
4. **Adding bias terms** — extend the model with user/item biases, compare RMSE and ranking metrics
5. **Evaluation** — HR@10, MRR@10, NDCG@10 using `metrics.py`, compare against MostPopular baseline
6. **Latent space visualization** — project item embeddings to 2D with PCA, visualize genre clustering

---

## Key Takeaways

- SVD decomposes R into user and item factor matrices — the best rank-k approximation
- Direct optimization via SGD handles sparsity cleanly — only observed ratings contribute to the loss
- Bias terms ($\mu$, $b_u$, $b_i$) are a cheap, high-value addition
- The learned embedding matrices P and Q are the direct ancestors of embedding layers in all neural RecSys models
- SGD-based MF is fast to train but sensitive to learning rate and initialization — use small random initialization
