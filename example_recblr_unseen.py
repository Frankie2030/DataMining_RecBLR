"""
Example: Using RecBLR with Unseen Item Handling

This script demonstrates how to train and evaluate RecBLR with preprocessing
and postprocessing for unseen items, following the Mamba4Rec approach on H&M.

The workflow:
1. Prepare dataset with RecBole
2. Load item features and create descriptions
3. Create and fit UnseenItemHandler
4. Train RecBLR (standard)
5. Enable unseen handling for evaluation
6. Evaluate with extended item catalog
"""

import os
import sys
import pandas as pd
import numpy as np
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed, set_color

from recblr_with_unseen import RecBLRWithUnseen, create_unseen_handler_from_dataset
from unseen_item_handler import create_item_descriptions_from_features


def main():
    """Main training and evaluation pipeline."""

    # ============================================================================
    # 1. Configuration
    # ============================================================================

    # Example config for H&M-style dataset
    config_dict = {
        'model': 'RecBLR',
        'dataset': 'hm',  # or your dataset name

        # RecBLR hyperparameters
        'hidden_size': 64,
        'num_layers': 2,
        'dropout_prob': 0.5,
        'expand': 2,
        'd_conv': 4,
        'bd_lru_only': False,
        'disable_conv1d': False,
        'disable_ffn': False,

        # Training
        'loss_type': 'CE',
        'epochs': 100,
        'train_batch_size': 2048,
        'eval_batch_size': 512,
        'learning_rate': 0.001,
        'eval_step': 1,

        # Data
        'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'user_inter_num_interval': '[5,inf)',
        'item_inter_num_interval': '[5,inf)',

        # Evaluation
        'metrics': ['Recall', 'NDCG', 'MRR'],
        'topk': [10, 20],
        'valid_metric': 'NDCG@10',
        'eval_args': {
            'split': {'RS': [0.8, 0.1, 0.1]},
            'mode': 'full',
            'order': 'TO'
        },

        # Other
        'seed': 42,
        'reproducibility': True,
        'data_path': 'dataset',
    }

    config = Config(model=RecBLRWithUnseen, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()

    logger.info("=" * 80)
    logger.info("RecBLR with Unseen Item Handling")
    logger.info("=" * 80)

    # ============================================================================
    # 2. Load Dataset
    # ============================================================================

    logger.info("Loading dataset...")
    dataset = create_dataset(config)
    logger.info(f"Dataset: {dataset}")
    logger.info(f"Training vocabulary: {dataset.item_num - 1} items")

    train_data, valid_data, test_data = data_preparation(config, dataset)

    # ============================================================================
    # 3. Create Unseen Item Handler
    # ============================================================================

    logger.info("\nCreating Unseen Item Handler...")

    # Example: Load item features from H&M dataset
    # Adjust paths and columns according to your dataset
    item_features_path = 'dataset/hm/articles.csv'

    if os.path.exists(item_features_path):
        logger.info(f"Loading item features from {item_features_path}")

        # Load item data
        item_data = pd.read_csv(
            item_features_path,
            dtype={'article_id': str}  # Ensure item IDs are strings
        )

        # Select features to use for item similarity
        # Adjust based on your dataset's columns
        feature_cols = [
            'prod_name',
            'product_type_name',
            'product_group_name',
            'colour_group_name',
            'perceived_colour_value_name',
            'perceived_colour_master_name',
            'department_name',
            'index_name',
            'section_name',
            'garment_group_name',
        ]

        # Filter to only use columns that exist
        feature_cols = [col for col in feature_cols if col in item_data.columns]

        logger.info(f"Using features: {feature_cols}")

        # Create item descriptions
        item_descriptions = create_item_descriptions_from_features(
            item_data,
            item_id_col='article_id',
            feature_cols=feature_cols
        )

        # Get valid items from training data
        valid_items = dataset.id2token(dataset.iid_field, range(1, dataset.item_num))

        # Create and fit handler
        from unseen_item_handler import UnseenItemHandler

        unseen_handler = UnseenItemHandler(
            item_descriptions=item_descriptions,
            valid_items=list(valid_items),
            n_components=16,  # PCA dimensionality
            random_state=config['seed']
        )
        unseen_handler.fit(verbose=True)

        # Optionally save the handler
        handler_path = 'saved/unseen_handler.pkl'
        os.makedirs('saved', exist_ok=True)
        unseen_handler.save(handler_path)
        logger.info(f"Saved UnseenItemHandler to {handler_path}")

    else:
        logger.warning(f"Item features not found at {item_features_path}")
        logger.warning("Unseen item handling will be disabled")
        unseen_handler = None

    # ============================================================================
    # 4. Train RecBLR
    # ============================================================================

    logger.info("\n" + "=" * 80)
    logger.info("Training RecBLR (standard training, no unseen handling)")
    logger.info("=" * 80)

    # Initialize model
    model = RecBLRWithUnseen(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # Initialize trainer
    trainer = Trainer(config, model)

    # Train
    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=True,
        show_progress=True
    )

    logger.info(set_color("Best valid result: ", "yellow") + f"{best_valid_result}")

    # ============================================================================
    # 5. Evaluate WITHOUT Unseen Handling (Baseline)
    # ============================================================================

    logger.info("\n" + "=" * 80)
    logger.info("Evaluation WITHOUT unseen handling (baseline)")
    logger.info("=" * 80)

    model.disable_unseen_handling()

    baseline_result = trainer.evaluate(test_data, show_progress=True)
    logger.info(set_color("Baseline test result: ", "yellow") + f"{baseline_result}")

    # ============================================================================
    # 6. Evaluate WITH Unseen Handling
    # ============================================================================

    if unseen_handler is not None:
        logger.info("\n" + "=" * 80)
        logger.info("Evaluation WITH unseen handling (preprocessing + postprocessing)")
        logger.info("=" * 80)

        # Enable unseen handling
        model.enable_unseen_handling(unseen_handler, verbose=True)

        # Note: RecBole's evaluation framework expects scores of shape (batch, n_items)
        # With unseen handling enabled, we produce (batch, total_catalog_size)
        # This might require custom evaluation - see below

        logger.info("\nTo properly evaluate with unseen items, you need:")
        logger.info("1. Test set with ground truth items (including unseen)")
        logger.info("2. Custom evaluation that handles extended score vectors")
        logger.info("\nSee the notebook example for detailed evaluation code.")

        # Example: Manual evaluation on a few sequences
        logger.info("\nExample predictions with unseen handling:")

        # Get a batch from test data
        for batch_idx, batched_data in enumerate(test_data):
            if batch_idx >= 1:  # Just show one batch
                break

            interaction = batched_data
            item_seq = interaction[model.ITEM_SEQ]
            item_seq_len = interaction[model.ITEM_SEQ_LEN]

            # Get predictions
            scores = model.full_sort_predict(interaction)

            logger.info(f"\nBatch {batch_idx}:")
            logger.info(f"  Input sequences: {item_seq.shape}")
            logger.info(f"  Sequence lengths: {item_seq_len}")
            logger.info(f"  Output scores shape: {scores.shape}")
            logger.info(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")

            # Get top-K predictions for first sequence
            topk = 20
            user_scores = scores[0].cpu().numpy()
            top_indices = np.argsort(-user_scores)[:topk]
            top_scores = user_scores[top_indices]

            logger.info(f"\n  Top-{topk} predictions for first user:")
            for rank, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
                # Map index to item token
                if idx < len(unseen_handler.item_inv_mapper):
                    item_token = unseen_handler.item_inv_mapper[idx]
                    is_valid = item_token in unseen_handler.valid_items
                    logger.info(
                        f"    {rank:2d}. Item {item_token} "
                        f"(score: {score:.4f}) "
                        f"{'[VALID]' if is_valid else '[UNSEEN]'}"
                    )

    # ============================================================================
    # 7. Summary
    # ============================================================================

    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)

    logger.info(set_color("Baseline (no unseen handling): ", "blue") + f"{baseline_result}")

    if unseen_handler is not None:
        coverage = len(unseen_handler.valid_items) / len(unseen_handler.item_mapper)
        logger.info(f"\nUnseen handling statistics:")
        logger.info(f"  - Training vocabulary: {len(unseen_handler.valid_items)} items")
        logger.info(f"  - Total catalog: {len(unseen_handler.item_mapper)} items")
        logger.info(f"  - Coverage: {coverage:.2%}")
        logger.info(f"  - Unseen items: {len(unseen_handler.item_mapper) - len(unseen_handler.valid_items)}")

    logger.info("\nDone!")


if __name__ == '__main__':
    main()
