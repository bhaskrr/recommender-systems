# Collaborative Filtering

To address some of the limitations of content-based filtering, collaborative filtering uses similarities between users and items simultaneously to provide recommendations.

This allows for serendipitous recommendations; that is, collaborative filtering models can recommend an item to user **A** based on the interests of a similar user **B**.

Furthermore, the embeddings can be learned automatically, without relying on hand-engineering of features.

## Types of Collaborative Filtering

Collaborative filtering (CF) methods are broadly categorized into two primary approaches: **Memory-Based** and **Model-Based**.

### Memory-Based Approaches

Memory-based approaches (also known as neighborhood-based or heuristic-based methods) use the entire historical user-item interaction dataset to generate recommendations.

1. **User-Based Collaborative Filtering**: This method recommends items by identifying users with tastes and preferences similar to the target user and suggesting items that those "neighbors" have liked but the target user has not yet experienced.

    Example: Social media "friend suggestions" are often based on this approach.
2. **Item-Based Collaborative Filtering**: This approach focuses on the relationships between items rather than users. It identifies items that are frequently liked by the same users and recommends one based on the other.

    Example: E-commerce sites like Amazon use the "users who bought this also bought" feature, which is a prime example of item-based filtering.

### Model-Based Approaches

Model-based approaches use machine learning and data mining algorithms to build a predictive model of user ratings or preferences from the interaction data. The goal is to discover underlying patterns or "latent factors" in the data.

Common techniques include:

1. **Matrix Factorization**: Methods like **Singular Value Decomposition (SVD)** decompose the large user-item matrix into smaller, lower-dimensional matrices that represent latent factors for users and items, which can then be used to predict missing ratings.

2. **Clustering Models**: Grouping similar users into clusters and using the cluster's collective preferences for recommendations.
3. **Deep Learning**: Utilizing neural networks and autoencoders to learn complex, non-linear patterns in user-item interactions.

## Steps of Collaborative Filtering

The steps of collaborative filtering generally follow a,memory-based or model-based approach, involving data collection, similarity calculation, and recommendation generation.

1. **Data Collection and Representation**: The foundation of CF is a User-Item Interaction Matrix.

   - Data Types:
      - Explicit Feedback: Direct user ratings (e.g., 1–5 stars).
      - Implicit Feedback: Observed behaviors like clicks, views, downloads, or purchase history.
   - Matrix Structure: Rows represent users, columns represent items and cells contain the interaction value.
   - Sparsity Challenge: In real-world systems, most cells are empty (null) because users only interact with a small fraction of available items.

2. **Data Preprocessing**: Raw data is cleaned to ensure consistency and improve model accuracy.

    - Handling Sparsity: Filling missing values with zeros or using dimensionality reduction to manage large, empty matrices.
    - Normalization:
        - Mean-Centering: Subtracting a user’s average rating from each of their ratings to account for "tough" vs. "generous" raters.
        - Z-Score: Adjusting for both mean and variance in user rating patterns.
    - Filtering: Removing users or items with too few interactions to provide reliable data.

3. **Similarity Computation**: The system determines "proximity" in vector space using mathematical metrics.

    - Approach Selection:
        - User-Based (UBCF): Finds users with similar rating patterns to the target user.
        - Item-Based (IBCF): Identifies items that have been rated similarly by the same group of users.
    - Key Metrics:
        - Cosine Similarity: Measures the angle between two vectors; values range from -1 to 1.
        - Pearson Correlation (PCC): Measures how two sets of ratings linearly relate, correcting for user bias.
        - Jaccard Similarity: Used primarily for binary (yes/no) interaction data.

4. **Prediction and Generation**:
Once similarities are calculated, the system predicts values for the "null" cells in the matrix.

    - Identify Neighbors: Select the top most similar users or items (K-Nearest Neighbors).
    - Weighted Aggregation: Calculate a predicted score by taking a weighted average of the neighbors' ratings, where the weight is the similarity score.
    - Top-N Selection:
        - Filter out items the user has already interacted with.
        - Sort remaining items by predicted score in descending order.
        - Present the top items to the user.

## Pros and Cons

### Pros

1. **No Domain Knowledge Needed**: The system automatically learns preferences, requiring only user interaction data (ratings, clicks, purchases) rather than detailed item content.
2. **Serendipity & Discovery**: It can recommend items outside a user’s historical preferences by identifying, for example, that users who liked A and B also liked C, allowing users to discover new interests.
3. **High-Quality Personalization**: Because it leverages the wisdom of the crowd, it can provide highly relevant, tailored recommendations that reflect niche, similar user groups.
4. **Effective Across Domains**: It can work with any type of item (movies, products, articles) without needing specialized feature extraction for each.

### Cons

1. **Cold-Start Problem**: New users or new items with no existing interaction history cannot be easily recommended, leading to poor initial performance.
2. Data Sparsity: In large catalogs, the number of ratings per item is often very low compared to the total items, making it difficult to find reliable, similar user matches.
3. **Scalability Issues**: As the number of users and items increases, the computational cost of calculating user-user or item-item similarities can become immense and slow.
4. **Popularity Bias**: The system tends to favor popular items, making it harder for obscure or new items to be recommended (the "first-rater" problem).
5. **"Gray Sheep"** Problem: Users with unique or highly diverse tastes that do not match any specific group of users are difficult to serve.
