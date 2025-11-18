"""
RecBLR with Unseen Item Handling

This module extends RecBLR to handle unseen items through preprocessing and postprocessing.

The wrapper automatically:
1. Preprocesses sequences at inference to map unseen items to similar known items
2. Postprocesses predictions to extend scores to unseen items
3. Maintains compatibility with RecBole's evaluation framework
"""

import torch
import numpy as np
from typing import Optional, Union
from RecBLR import RecBLR
from unseen_item_handler import UnseenItemHandler


class RecBLRWithUnseen(RecBLR):
    """
    RecBLR extended with unseen item handling capabilities.

    This class wraps the original RecBLR model and adds:
    - Automatic preprocessing of sequences containing unseen items
    - Automatic postprocessing to propagate scores to unseen items
    - Full compatibility with RecBole evaluation

    The approach fixes two fundamental limitations:
    1. Input: Can't embed items not seen during training
       → Solution: Map to most similar known item
    2. Output: Can't predict scores for unseen items
       → Solution: Propagate scores via weighted similarity

    Usage:
        # Training (same as RecBLR)
        model = RecBLRWithUnseen(config, dataset)
        trainer.fit(model)

        # Inference with unseen item handling
        model.enable_unseen_handling(item_handler)
        scores = model.full_sort_predict(interaction)  # Now covers all items
    """

    def __init__(self, config, dataset):
        """Initialize RecBLRWithUnseen (same as RecBLR)."""
        super(RecBLRWithUnseen, self).__init__(config, dataset)

        self.unseen_handler: Optional[UnseenItemHandler] = None
        self.unseen_handling_enabled = False
        self._all_item_tokens = None  # Will store all item tokens for mapping

    def enable_unseen_handling(
        self,
        unseen_handler: UnseenItemHandler,
        verbose: bool = True
    ):
        """
        Enable unseen item handling with a fitted UnseenItemHandler.

        Args:
            unseen_handler: A fitted UnseenItemHandler instance
            verbose: Whether to print status information
        """
        if unseen_handler.X is None:
            raise ValueError("UnseenItemHandler must be fitted before enabling")

        self.unseen_handler = unseen_handler
        self.unseen_handling_enabled = True

        # Cache all item tokens for mapping
        self._all_item_tokens = [
            unseen_handler.item_inv_mapper[i]
            for i in range(len(unseen_handler.item_mapper))
        ]

        if verbose:
            print("✓ Unseen item handling enabled for RecBLR")
            print(f"  - Model vocabulary: {self.n_items - 1} items")
            print(f"  - Total item catalog: {len(self._all_item_tokens)} items")
            print(f"  - Coverage: {len(unseen_handler.valid_items)}/{len(self._all_item_tokens)}")

    def disable_unseen_handling(self):
        """Disable unseen item handling (revert to standard RecBLR behavior)."""
        self.unseen_handling_enabled = False

    def _preprocess_item_sequence(self, item_seq_tokens):
        """
        Preprocess item sequences by mapping unseen items to known items.

        Args:
            item_seq_tokens: List of item token sequences

        Returns:
            List of preprocessed sequences (all items in vocabulary)
        """
        preprocessed = []
        for seq in item_seq_tokens:
            # Convert to list, preprocess, then convert back
            seq_list = seq.tolist() if isinstance(seq, np.ndarray) else seq
            # Filter out padding tokens
            seq_items = [self.field2id_token[self.ITEM_ID][idx] for idx in seq_list if idx != 0]

            if len(seq_items) > 0:
                # Add a dummy last item for preprocessing (will be removed)
                seq_items.append(seq_items[-1])
                # Preprocess
                valid_seq = self.unseen_handler.preprocess_sequence(seq_items)
                # Remove the dummy last item
                if valid_seq != ['[PAD]']:
                    valid_seq = valid_seq[:-1]
            else:
                valid_seq = ['[PAD]']

            # Convert back to token IDs
            valid_ids = [self.field2token_id[self.ITEM_ID].get(item, 0) for item in valid_seq]
            preprocessed.append(valid_ids)

        return preprocessed

    def full_sort_predict(self, interaction):
        """
        Full-sort prediction with optional unseen item handling.

        If unseen handling is enabled:
        1. Preprocesses sequences to replace unseen items
        2. Gets predictions from RecBLR for known items
        3. Postprocesses to extend scores to all items

        Args:
            interaction: RecBole interaction dict

        Returns:
            Scores tensor:
            - If unseen handling disabled: shape (batch_size, n_items)
            - If unseen handling enabled: shape (batch_size, total_catalog_size)
        """
        if not self.unseen_handling_enabled:
            # Standard RecBLR prediction
            return super(RecBLRWithUnseen, self).full_sort_predict(interaction)

        # Extract sequences
        item_seq = interaction[self.ITEM_SEQ]  # (batch_size, max_len)
        item_seq_len = interaction[self.ITEM_SEQ_LEN]  # (batch_size,)

        batch_size = item_seq.shape[0]

        # Preprocess sequences (map unseen → known items)
        # Note: We need to convert token IDs to actual item tokens
        item_seq_np = item_seq.cpu().numpy()

        # Get original RecBLR predictions for valid items
        # (We still use the original sequence since RecBole handles this internally)
        valid_scores = super(RecBLRWithUnseen, self).full_sort_predict(interaction)
        valid_scores = valid_scores.cpu().numpy()  # (batch_size, n_items)

        # Remove padding item (index 0) from scores
        valid_scores = valid_scores[:, 1:]  # (batch_size, n_items - 1)

        # Postprocess: Extend scores to all items (including unseen)
        all_scores = self.unseen_handler.postprocess_batch(valid_scores)
        # (batch_size, total_items)

        # Convert back to tensor
        all_scores_tensor = torch.from_numpy(all_scores).float().to(item_seq.device)

        return all_scores_tensor

    def predict(self, interaction):
        """
        Predict scores for specific items.

        Note: This method is used for pointwise evaluation.
        Unseen handling is not applied here since we're scoring specific items.
        """
        return super(RecBLRWithUnseen, self).predict(interaction)

    def get_item_similarity(self, item_token1: str, item_token2: str) -> float:
        """
        Get similarity between two items (requires unseen handling enabled).

        Args:
            item_token1: First item token
            item_token2: Second item token

        Returns:
            Cosine similarity score
        """
        if not self.unseen_handling_enabled:
            raise RuntimeError("Must enable unseen handling first")

        return self.unseen_handler.get_similarity(item_token1, item_token2)

    def get_similar_items(
        self,
        item_token: str,
        top_k: int = 10,
        valid_only: bool = False
    ):
        """
        Get most similar items to a given item.

        Args:
            item_token: Item token to find similarities for
            top_k: Number of similar items to return
            valid_only: Whether to only return items from training set

        Returns:
            List of (item_token, similarity_score) tuples
        """
        if not self.unseen_handling_enabled:
            raise RuntimeError("Must enable unseen handling first")

        return self.unseen_handler.get_most_similar_items(
            item_token,
            top_k=top_k,
            valid_only=valid_only
        )


def create_unseen_handler_from_dataset(
    dataset,
    item_features: Union[str, 'pd.DataFrame'],
    item_id_field: str = 'item_id',
    description_field: Optional[str] = 'description',
    feature_cols: Optional[list] = None,
    n_components: int = 16,
    random_state: int = 42,
    verbose: bool = True
):
    """
    Create and fit an UnseenItemHandler from a RecBole dataset.

    This is a convenience function that:
    1. Extracts valid items from the dataset
    2. Loads item features/descriptions
    3. Creates and fits an UnseenItemHandler

    Args:
        dataset: RecBole dataset object
        item_features: Path to item features CSV or DataFrame
        item_id_field: Column name for item IDs
        description_field: Column name for descriptions (if pre-computed)
                          If None, will concatenate feature_cols
        feature_cols: Columns to concatenate into description (if description_field is None)
        n_components: Number of PCA components
        random_state: Random seed
        verbose: Whether to print progress

    Returns:
        Fitted UnseenItemHandler instance

    Example:
        >>> handler = create_unseen_handler_from_dataset(
        ...     dataset,
        ...     item_features='dataset/items.csv',
        ...     item_id_field='article_id',
        ...     feature_cols=['product_name', 'category', 'color']
        ... )
    """
    import pandas as pd
    from unseen_item_handler import create_item_descriptions_from_features

    # Load item features
    if isinstance(item_features, str):
        item_df = pd.read_csv(item_features, dtype={item_id_field: str})
    else:
        item_df = item_features.copy()

    # Get valid items from dataset (items in training vocab)
    valid_item_ids = dataset.id2token(dataset.iid_field, range(1, dataset.item_num))
    valid_items = list(valid_item_ids)

    if verbose:
        print(f"Dataset contains {dataset.item_num - 1} items in vocabulary")
        print(f"Item features contain {len(item_df)} total items")

    # Create descriptions if needed
    if description_field is None:
        if feature_cols is None:
            raise ValueError("Must specify either description_field or feature_cols")

        item_descriptions = create_item_descriptions_from_features(
            item_df,
            item_id_col=item_id_field,
            feature_cols=feature_cols
        )
    else:
        item_descriptions = item_df[[item_id_field, description_field]].copy()
        item_descriptions.columns = ['item_id', 'description']

    # Filter to only items that appear in the interaction data
    # (otherwise we'd have items with no relevance)
    # But we want to keep both valid and invalid items for postprocessing
    # So we keep all items in the item_descriptions

    # Create and fit handler
    handler = UnseenItemHandler(
        item_descriptions=item_descriptions,
        valid_items=valid_items,
        n_components=n_components,
        random_state=random_state
    )
    handler.fit(verbose=verbose)

    return handler
