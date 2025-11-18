"""
Evaluation Script for RecBLR with Unseen Items

This script demonstrates how to evaluate RecBLR on test sequences that contain
unseen items, following the approach from Mamba4Rec on H&M dataset.

It computes:
1. NDCG with preprocessing only (unseen → similar known items in input)
2. NDCG with pre + postprocessing (also propagate scores to unseen items)
"""

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import ndcg_score
from functools import lru_cache
from tqdm import tqdm


def evaluate_with_unseen_items(
    model,
    test_sequences,
    unseen_handler,
    dataset,
    device='cuda',
    verbose=True
):
    """
    Evaluate RecBLR on test sequences containing unseen items.

    Args:
        model: Trained RecBLR or RecBLRWithUnseen model
        test_sequences: DataFrame with columns ['user_id', 'sequence']
                       where 'sequence' is a list of item tokens
        unseen_handler: Fitted UnseenItemHandler instance
        dataset: RecBole dataset object
        device: Device to run model on
        verbose: Whether to print progress

    Returns:
        Dictionary with evaluation metrics
    """

    if verbose:
        print("Evaluating RecBLR with unseen item handling...")
        print(f"Test sequences: {len(test_sequences)}")

    model.eval()
    model = model.to(device)

    # Prepare test data
    test_df = test_sequences.copy()

    # Create mapping function (cached)
    @lru_cache(maxsize=2048)
    def convert2valid(item):
        """Convert unseen item to most similar valid item."""
        if item in unseen_handler.valid_items:
            return item
        item_id = unseen_handler.item_mapper[item]
        sim_scores = unseen_handler.sim_cosine_valid[item_id]
        most_similar_idx = np.argmax(sim_scores)
        return unseen_handler.valid_inv_mapper[most_similar_idx]

    # Preprocess sequences: map unseen items to valid items
    def to_valid_list(item_list):
        """Convert sequence to valid items only."""
        valid_list = []
        for item in item_list[:-1]:  # Exclude last item (target)
            if item not in unseen_handler.valid_items:
                valid_list.append(convert2valid(item))
            else:
                valid_list.append(item)
        return ['[PAD]'] if len(valid_list) == 0 else valid_list

    if verbose:
        print("Preprocessing sequences...")

    test_df['item_id_list'] = test_df['sequence'].apply(to_valid_list)
    test_df['item_length'] = test_df['item_id_list'].apply(len)

    # ============================================================================
    # Evaluation 1: With Preprocessing Only (Hit NDCG)
    # ============================================================================

    if verbose:
        print("\nEvaluation 1: Preprocessing only (mapping unseen → valid)")

    shape = (len(test_df), dataset.item_num - 1)
    y_scores = np.zeros(shape)
    y_true = np.zeros(shape)

    with torch.no_grad():
        iterator = enumerate(test_df.iterrows())
        if verbose:
            iterator = tqdm(list(iterator), desc="Computing predictions")

        for idx, (_, row) in iterator:
            # Convert item tokens to IDs
            item_tokens = row['item_id_list']
            item_ids = [
                dataset.token2id(dataset.iid_field, token)
                for token in item_tokens
            ]

            # Create interaction
            max_len = 50  # Should match model's max_seq_length
            item_seq = np.zeros(max_len, dtype=np.int64)
            seq_len = min(len(item_ids), max_len)
            if seq_len > 0:
                item_seq[-seq_len:] = item_ids[-seq_len:]

            interaction = {
                'item_id_list': torch.LongTensor([item_seq]).to(device),
                'item_length': torch.LongTensor([seq_len]).to(device)
            }

            # Get predictions
            scores = model.full_sort_predict(interaction)[0]
            y_scores[idx] = scores[1:].cpu().numpy()  # Exclude padding

            # Ground truth
            true_item = row['sequence'][-1]
            if true_item in unseen_handler.valid_items:
                # Map to valid item for ground truth
                true_valid = true_item
            else:
                # Map unseen item to most similar valid
                true_valid = convert2valid(true_item)

            true_id = dataset.token2id(dataset.iid_field, true_valid) - 1
            if true_id >= 0:
                y_true[idx, true_id] = 1

    # Compute NDCG
    ndcg_preprocessing = ndcg_score(y_true, y_scores)

    if verbose:
        print(f"✓ NDCG@All with preprocessing: {ndcg_preprocessing:.4f}")

    # ============================================================================
    # Evaluation 2: With Pre + Postprocessing (Extended Catalog)
    # ============================================================================

    if verbose:
        print("\nEvaluation 2: Preprocessing + Postprocessing (full catalog)")

    # Prepare similarity weights for postprocessing
    non_valid_ids = [
        unseen_handler.item_mapper[item]
        for item in unseen_handler.item_mapper.keys()
        if item not in unseen_handler.valid_items
    ]
    valid_ids = [
        unseen_handler.item_mapper[item]
        for item in unseen_handler.valid_items
    ]

    sim_scores = unseen_handler.sim_cosine_valid[non_valid_ids]
    sim_weights = sim_scores / np.sum(sim_scores, axis=1, keepdims=True)

    def weighted_similarity(valid_scores):
        """Propagate scores from valid to all items."""
        final_scores = np.zeros(len(unseen_handler.item_mapper))
        final_scores[valid_ids] = valid_scores
        final_scores[non_valid_ids] = np.dot(sim_weights, valid_scores)
        return final_scores

    # Evaluate with postprocessing
    post_shape = (len(test_df), len(unseen_handler.item_mapper))
    y_scores_post = np.zeros(post_shape)
    y_true_post = np.zeros(post_shape)

    with torch.no_grad():
        iterator = enumerate(test_df.iterrows())
        if verbose:
            iterator = tqdm(list(iterator), desc="Computing predictions")

        for idx, (_, row) in iterator:
            # Get predictions for valid items (same as before)
            item_tokens = row['item_id_list']
            item_ids = [
                dataset.token2id(dataset.iid_field, token)
                for token in item_tokens
            ]

            max_len = 50
            item_seq = np.zeros(max_len, dtype=np.int64)
            seq_len = min(len(item_ids), max_len)
            if seq_len > 0:
                item_seq[-seq_len:] = item_ids[-seq_len:]

            interaction = {
                'item_id_list': torch.LongTensor([item_seq]).to(device),
                'item_length': torch.LongTensor([seq_len]).to(device)
            }

            scores = model.full_sort_predict(interaction)[0]
            valid_scores = scores[1:].cpu().numpy()

            # Postprocess: extend to all items
            y_scores_post[idx] = weighted_similarity(valid_scores)

            # Ground truth (true item, potentially unseen)
            true_item = row['sequence'][-1]
            true_item_id = unseen_handler.item_mapper[true_item]
            y_true_post[idx, true_item_id] = 1

    # Compute NDCG
    ndcg_postprocessing = ndcg_score(y_true_post, y_scores_post)

    if verbose:
        print(f"✓ NDCG@All with pre+postprocessing: {ndcg_postprocessing:.4f}")

    # ============================================================================
    # Evaluation 3: Similarity-based NDCG (relaxed metric)
    # ============================================================================

    if verbose:
        print("\nEvaluation 3: Similarity-based NDCG (relaxed metric)")

    # For preprocessing only
    y_true_sim = np.zeros(shape)
    for idx, row in test_df.iterrows():
        true_item = row['sequence'][-1]
        true_item_id = unseen_handler.item_mapper[true_item]
        # Consider items with high similarity (>0.9) as relevant
        similarities = unseen_handler.sim_cosine_valid[true_item_id, :]
        y_true_sim[idx] = (similarities > 0.9).astype(int)

    ndcg_sim_pre = ndcg_score(y_true_sim, y_scores)

    # For pre + postprocessing
    y_true_sim_post = np.zeros(post_shape)
    for idx, row in test_df.iterrows():
        true_item = row['sequence'][-1]
        true_item_id = unseen_handler.item_mapper[true_item]
        similarities = unseen_handler.sim_cosine_all[true_item_id, :]
        y_true_sim_post[idx] = (similarities > 0.9).astype(int)

    ndcg_sim_post = ndcg_score(y_true_sim_post, y_scores_post)

    if verbose:
        print(f"✓ Similarity NDCG with preprocessing: {ndcg_sim_pre:.4f}")
        print(f"✓ Similarity NDCG with pre+post: {ndcg_sim_post:.4f}")

    # ============================================================================
    # Summary
    # ============================================================================

    results = {
        'ndcg_preprocessing_only': ndcg_preprocessing,
        'ndcg_pre_and_post': ndcg_postprocessing,
        'ndcg_similarity_pre': ndcg_sim_pre,
        'ndcg_similarity_post': ndcg_sim_post,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Hit NDCG with preprocessing:           {results['ndcg_preprocessing_only']:.4f}")
        print(f"Hit NDCG with pre+post:                {results['ndcg_pre_and_post']:.4f}")
        print(f"Similarity NDCG with preprocessing:    {results['ndcg_similarity_pre']:.4f}")
        print(f"Similarity NDCG with pre+post:         {results['ndcg_similarity_post']:.4f}")
        print("=" * 60)

    return results


# Example usage
if __name__ == '__main__':
    print("""
    Example Usage:
    --------------

    # 1. Load your trained RecBLR model
    from RecBLR import RecBLR
    model = RecBLR(config, dataset)
    model.load_state_dict(torch.load('saved/model.pth'))

    # 2. Load test sequences (with unseen items)
    test_sequences = pd.DataFrame({
        'user_id': ['user1', 'user2', ...],
        'sequence': [['item1', 'item2', 'target1'], ['item3', 'item4', 'target2'], ...]
    })

    # 3. Load or create UnseenItemHandler
    from unseen_item_handler import UnseenItemHandler
    unseen_handler = UnseenItemHandler.load('saved/unseen_handler.pkl')

    # 4. Evaluate
    results = evaluate_with_unseen_items(
        model=model,
        test_sequences=test_sequences,
        unseen_handler=unseen_handler,
        dataset=dataset,
        device='cuda'
    )

    print(results)
    """)
