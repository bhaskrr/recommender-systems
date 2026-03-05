# Content-based Filtering

Content-based filtering uses item features to recommend other items similar to what the user likes, based on their previous actions or explicit feedback.

Similarities are calculated over product metadata, and it provides the opportunity to develop recommendations. The products that are most similar to the relevant product are recommended.

Metadata represents the features of a product/service. For example, the director, cast, and screenwriter of a movie; the author, back cover article, translator of a book, or category information of a product.

## Steps of Content-Based Filtering

1. **Item Feature Extraction (Item Representation)**:
Identify and extract relevant features from items, such as keywords, genre, director, or description for movies, or text content for articles.
This converts raw content into a structured format (e.g., vectors of keywords or genres).

2. **User Profile Creation**:
Build a profile representing a user's preferences based on the features of items they have previously interacted with, rated, or liked.
This profile is often a weighted vector representing the importance of different features to the user.

3. **Similarity Computation (Matching)**:
Compare the user profile against the features of items not yet seen by the user.
Common techniques to measure this similarity include **Cosine Similarity, Euclidean Distance, or Dot Product**.

4. **Recommendation Generation**:
Rank the items based on their similarity scores to the user profile.
Suggest the items with the highest scores.

## Pros and Cons

### Pros

1. The model doesn't need any data about other users, since the recommendations are specific to this user. This makes it easier to scale to a large number of users.
2. The model can capture the specific interests of a user, and can recommend niche items that very few other users are interested in.

### Cons

1. Since the feature representation of the items are hand-engineered to some extent, this technique requires a lot of domain knowledge. Therefore, the model can only be as good as the hand-engineered features.
2. The model can only make recommendations based on existing interests of the user. In other words, the model has limited ability to expand on the users' existing interests.
