"""
Unseen Item Handler for Sequential Recommender Systems

This module provides utilities to handle unseen items (items that appear in test/validation
but not in training) for embedding-based sequential recommenders like RecBLR, Mamba4Rec, etc.

The approach follows the methodology from Mamba4Rec applied to H&M dataset:
1. Preprocessing: Map unseen items to their most similar known items based on content similarity
2. Postprocessing: Propagate scores from known items to unseen items using weighted similarity

This addresses the fundamental limitation that embedding-based models can only:
- Embed items seen during training
- Produce predictions for items in the training vocabulary
"""

import numpy as np
import pandas as pd
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from typing import Dict, List, Optional, Tuple, Union
import pickle


class UnseenItemHandler:
    """
    Handles preprocessing and postprocessing for unseen items in sequential recommenders.

    Key functionality:
    - Computes item similarity based on textual features (TF-IDF + PCA)
    - Maps unseen items to most similar known items (preprocessing)
    - Propagates scores from known to unseen items (postprocessing)

    Attributes:
        item_descriptions: DataFrame with item IDs and textual descriptions
        valid_items: Set of items that appeared in training data
        n_components: Number of PCA components for dimensionality reduction
        similarity_threshold: Optional threshold for considering items similar
    """

    def __init__(
        self,
        item_descriptions: pd.DataFrame,
        valid_items: List[str],
        n_components: int = 16,
        random_state: int = 42,
        cache_size: int = 2048
    ):
        """
        Initialize the UnseenItemHandler.

        Args:
            item_descriptions: DataFrame with columns ['item_id', 'description']
            valid_items: List of item IDs that appeared in training
            n_components: Number of components for TruncatedSVD dimensionality reduction
            random_state: Random seed for reproducibility
            cache_size: Size of LRU cache for item mapping
        """
        self.item_descriptions = item_descriptions.copy()
        self.valid_items = set(valid_items)
        self.n_components = n_components
        self.random_state = random_state
        self.cache_size = cache_size

        # Will be populated during fit()
        self.item_mapper = None
        self.item_inv_mapper = None
        self.valid_mapper = None
        self.valid_inv_mapper = None
        self.X = None  # Item feature vectors (all items)
        self.X_valid = None  # Item feature vectors (valid items only)
        self.sim_cosine_all = None  # Similarity matrix (all items)
        self.sim_cosine_valid = None  # Similarity matrix (all vs valid)
        self.sim_weights = None  # Normalized similarity weights for postprocessing

        self._convert2valid = None  # Cached conversion function

    def fit(self, verbose: bool = True):
        """
        Fit the similarity model on item descriptions.

        This computes:
        1. TF-IDF vectors from item descriptions
        2. Dimensionality reduction via TruncatedSVD
        3. Cosine similarity matrices
        4. Normalized similarity weights for score propagation

        Args:
            verbose: Whether to print progress information
        """
        if verbose:
            print("Fitting UnseenItemHandler...")
            print(f"Total items: {len(self.item_descriptions)}")
            print(f"Valid items (in training): {len(self.valid_items)}")

        # Sort by item_id for consistent indexing
        self.item_descriptions = self.item_descriptions.sort_values('item_id').reset_index(drop=True)

        # Create bidirectional mappers
        self.item_mapper = {
            self.item_descriptions['item_id'].iloc[i]: i
            for i in range(len(self.item_descriptions))
        }
        self.item_inv_mapper = {
            i: self.item_descriptions['item_id'].iloc[i]
            for i in range(len(self.item_descriptions))
        }

        if verbose:
            print("Computing TF-IDF vectors...")

        # Compute TF-IDF vectors
        vect = TfidfVectorizer()
        tfidf = vect.fit_transform(self.item_descriptions['description'])

        if verbose:
            print(f"TF-IDF vector size: {tfidf.shape[1]}")
            print(f"Reducing to {self.n_components} components with TruncatedSVD...")

        # Dimensionality reduction
        X = csr_matrix(tfidf)
        svd = TruncatedSVD(
            n_components=self.n_components,
            n_iter=3,
            random_state=self.random_state
        )
        self.X = svd.fit_transform(X)

        if verbose:
            print(f"Reduced vector size: {self.X.shape[1]}")
            print("Preparing valid item mappings...")

        # Get valid item indices
        valid_articles = [item for item in self.item_descriptions['item_id'] if item in self.valid_items]
        valid_ids = [self.item_mapper[item] for item in valid_articles]
        self.X_valid = self.X[valid_ids]

        self.valid_mapper = {valid_articles[i]: i for i in range(len(valid_articles))}
        self.valid_inv_mapper = {i: valid_articles[i] for i in range(len(valid_articles))}

        if verbose:
            print("Computing cosine similarity matrices...")

        # Compute similarity matrices
        self.sim_cosine_all = cosine_similarity(self.X, self.X)
        self.sim_cosine_valid = cosine_similarity(self.X, self.X_valid)

        if verbose:
            print("Preparing similarity weights for postprocessing...")

        # Prepare normalized similarity weights for postprocessing
        non_valid_ids = [
            self.item_mapper[item]
            for item in self.item_mapper.keys()
            if item not in self.valid_items
        ]

        # Normalize similarities to sum to 1 for each unseen item
        sim_scores = self.sim_cosine_valid[non_valid_ids]
        self.sim_weights = sim_scores / np.sum(sim_scores, axis=1, keepdims=True)
        self._non_valid_ids = non_valid_ids
        self._valid_ids = valid_ids

        # Create cached conversion function
        self._create_convert2valid_function()

        if verbose:
            print("✓ UnseenItemHandler fitted successfully!")
            print(f"  - Coverage: {len(self.valid_items)}/{len(self.item_descriptions)} items in training")
            print(f"  - Unseen items: {len(non_valid_ids)}")

    def _create_convert2valid_function(self):
        """Create a cached function for mapping unseen items to valid items."""
        @lru_cache(maxsize=self.cache_size)
        def convert2valid(item):
            """Convert an unseen item to its most similar valid item."""
            item_id = self.item_mapper[item]
            sim_scores = self.sim_cosine_valid[item_id]
            most_similar_idx = np.argmax(sim_scores)
            return self.valid_inv_mapper[most_similar_idx]

        self._convert2valid = convert2valid

    def preprocess_sequence(
        self,
        item_sequence: List[str],
        padding_token: str = '[PAD]'
    ) -> List[str]:
        """
        Preprocess a sequence by mapping unseen items to their most similar valid items.

        This is the PREPROCESSING step that fixes the input side:
        - Unseen items are replaced with their most similar known items
        - The sequence becomes fully in-vocabulary
        - RecBLR can now process a coherent sequence

        Args:
            item_sequence: List of item IDs (may contain unseen items)
            padding_token: Token to use if sequence becomes empty

        Returns:
            List of valid item IDs (all known to the model)
        """
        if self._convert2valid is None:
            raise RuntimeError("Must call fit() before preprocess_sequence()")

        valid_list = []

        # Process all items except the last (which is the target)
        for item in item_sequence[:-1]:
            if item not in self.valid_items:
                # Map to most similar valid item
                valid_list.append(self._convert2valid(item))
            else:
                # Already valid
                valid_list.append(item)

        return [padding_token] if len(valid_list) == 0 else valid_list

    def preprocess_batch(
        self,
        sequences: List[List[str]],
        padding_token: str = '[PAD]'
    ) -> List[List[str]]:
        """
        Preprocess a batch of sequences.

        Args:
            sequences: List of item sequences
            padding_token: Token to use for empty sequences

        Returns:
            List of preprocessed sequences
        """
        return [self.preprocess_sequence(seq, padding_token) for seq in sequences]

    def postprocess_scores(
        self,
        valid_item_scores: np.ndarray
    ) -> np.ndarray:
        """
        Postprocess scores by propagating them to unseen items.

        This is the POSTPROCESSING step that fixes the output side:
        - Model produces scores only for known items
        - We propagate these scores to unseen items via weighted similarity
        - Final ranking covers the entire catalog

        The formula is:
            score(u_i) = Σ_j [sim(u_i, v_j) × score(v_j)] / Σ_k sim(u_i, v_k)

        Args:
            valid_item_scores: Scores for valid items, shape (n_valid_items,)

        Returns:
            Scores for all items, shape (n_total_items,)
        """
        if self.sim_weights is None:
            raise RuntimeError("Must call fit() before postprocess_scores()")

        n_total_items = self.X.shape[0]
        final_scores = np.zeros(n_total_items)

        # Keep original scores for valid items
        final_scores[self._valid_ids] = valid_item_scores

        # Propagate scores to unseen items using weighted similarity
        final_scores[self._non_valid_ids] = np.dot(self.sim_weights, valid_item_scores)

        return final_scores

    def postprocess_batch(
        self,
        valid_scores_batch: np.ndarray
    ) -> np.ndarray:
        """
        Postprocess a batch of score vectors.

        Args:
            valid_scores_batch: Batch of scores for valid items, shape (batch_size, n_valid_items)

        Returns:
            Batch of scores for all items, shape (batch_size, n_total_items)
        """
        batch_size = valid_scores_batch.shape[0]
        n_total_items = self.X.shape[0]
        final_scores = np.zeros((batch_size, n_total_items))

        for i in range(batch_size):
            final_scores[i] = self.postprocess_scores(valid_scores_batch[i])

        return final_scores

    def save(self, filepath: str):
        """Save the fitted handler to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str):
        """Load a fitted handler from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def get_similarity(self, item1: str, item2: str) -> float:
        """Get cosine similarity between two items."""
        idx1 = self.item_mapper[item1]
        idx2 = self.item_mapper[item2]
        return self.sim_cosine_all[idx1, idx2]

    def get_most_similar_items(
        self,
        item: str,
        top_k: int = 10,
        valid_only: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Get most similar items to a given item.

        Args:
            item: Item ID to find similarities for
            top_k: Number of similar items to return
            valid_only: Whether to only return valid (training) items

        Returns:
            List of (item_id, similarity_score) tuples
        """
        item_id = self.item_mapper[item]

        if valid_only:
            sim_scores = self.sim_cosine_valid[item_id]
            sort_indices = np.argsort(-sim_scores)[:top_k]
            return [
                (self.valid_inv_mapper[idx], sim_scores[idx])
                for idx in sort_indices
            ]
        else:
            sim_scores = self.sim_cosine_all[item_id]
            sort_indices = np.argsort(-sim_scores)[:top_k + 1]  # +1 to exclude self
            results = []
            for idx in sort_indices:
                if idx != item_id:  # Skip the item itself
                    results.append((self.item_inv_mapper[idx], sim_scores[idx]))
                if len(results) == top_k:
                    break
            return results


def create_item_descriptions_from_features(
    item_data: pd.DataFrame,
    item_id_col: str,
    feature_cols: Optional[List[str]] = None,
    exclude_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create item descriptions by concatenating textual features.

    This is a helper function to prepare the input format needed by UnseenItemHandler.

    Args:
        item_data: DataFrame containing item features
        item_id_col: Name of the column containing item IDs
        feature_cols: Specific columns to use (if None, uses all object/string columns)
        exclude_cols: Columns to exclude from description

    Returns:
        DataFrame with columns ['item_id', 'description']

    Example:
        >>> item_data = pd.DataFrame({
        ...     'article_id': ['001', '002'],
        ...     'product_name': ['Strap top', 'Stockings'],
        ...     'product_type': ['Vest top', 'Underwear'],
        ...     'color': ['White', 'Black'],
        ...     'price': [19.99, 9.99]
        ... })
        >>> descriptions = create_item_descriptions_from_features(
        ...     item_data,
        ...     item_id_col='article_id',
        ...     exclude_cols=['price']
        ... )
    """
    df = item_data.copy()

    # Select feature columns
    if feature_cols is not None:
        text_cols = feature_cols
    else:
        # Use all object/string columns
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        text_cols.remove(item_id_col)

    # Exclude specified columns
    if exclude_cols is not None:
        text_cols = [col for col in text_cols if col not in exclude_cols]

    # Concatenate features into description
    df['description'] = df[text_cols].apply(
        lambda x: ' '.join(x.dropna().astype(str)),
        axis=1
    )

    return df[[item_id_col, 'description']].rename(columns={item_id_col: 'item_id'})
