"""
Run RecBLR with unseen item handling (preprocessing and/or postprocessing).

This script extends the standard run.py to support evaluation with:
- Preprocessing only: Maps unseen items in input to similar known items
- Postprocessing only: Extends prediction scores to unseen items
- Both: Full preprocessing + postprocessing pipeline

Usage:
    python run_with_unseen.py --model R --mode preprocessing
    python run_with_unseen.py --model R --mode postprocessing
    python run_with_unseen.py --model R --mode both
"""

import sys
import logging
from logging import getLogger
import argparse
import os
import pandas as pd

from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from RecBLR import RecBLR
from recblr_with_unseen import RecBLRWithUnseen
from recbole.model.sequential_recommender.bert4rec import BERT4Rec
from recbole.model.sequential_recommender.sasrec import SASRec
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)
from plot_utils import parse_log_text, generate_plots
from unseen_item_handler import UnseenItemHandler


def create_item_features_for_dataset(dataset_name, dataset_path='dataset'):
    """
    Create item features DataFrame for common datasets.

    For datasets without item features, we create synthetic descriptions
    based on item ID and frequency.

    Args:
        dataset_name: Name of the dataset (e.g., 'ml-1m', 'yelp', 'amazon-beauty')
        dataset_path: Path to dataset directory

    Returns:
        DataFrame with columns ['item_id', 'description']
    """
    print(f"Loading item features for {dataset_name}...")

    # Check for pre-created features file
    features_file = os.path.join(dataset_path, dataset_name, f'{dataset_name}_item_features.csv')
    if os.path.exists(features_file):
        print(f"Found pre-created features: {features_file}")
        return pd.read_csv(features_file)

    # Try to load item file
    item_file = os.path.join(dataset_path, dataset_name, f'{dataset_name}.item')
    if os.path.exists(item_file):
        try:
            item_df = pd.read_csv(item_file, sep='\t')
            # Check column names (RecBole uses :token suffix)
            cols = item_df.columns.tolist()

            # Find item_id column
            item_id_col = None
            for col in cols:
                if 'item' in col.lower() and 'id' in col.lower():
                    item_id_col = col
                    break

            if item_id_col is None:
                print(f"Could not find item_id column in {item_file}")
            else:
                # Use available text columns as description
                text_cols = item_df.select_dtypes(include=['object']).columns.tolist()
                if item_id_col in text_cols:
                    text_cols.remove(item_id_col)

                if text_cols:
                    print(f"Using item features from {len(text_cols)} columns: {text_cols}")
                    item_df['description'] = item_df[text_cols].apply(
                        lambda x: ' '.join(x.dropna().astype(str)),
                        axis=1
                    )
                    result = item_df[[item_id_col, 'description']].copy()
                    result.columns = ['item_id', 'description']
                    return result
        except Exception as e:
            print(f"Could not load item features from {item_file}: {e}")

    # Try to create features from interactions
    inter_file = os.path.join(dataset_path, dataset_name, f'{dataset_name}.inter')
    if os.path.exists(inter_file):
        print(f"Creating synthetic features from interaction data: {inter_file}")
        try:
            from prepare_item_features import create_interaction_based_features

            # Create features
            output_file = features_file
            sample_df = pd.read_csv(inter_file, sep='\t', nrows=5)
            cols = sample_df.columns.tolist()

            # Find item column
            item_col = None
            user_col = None
            for col in cols:
                if 'item' in col.lower():
                    item_col = col
                if 'user' in col.lower():
                    user_col = col

            if item_col:
                features = create_interaction_based_features(
                    inter_file,
                    output_file,
                    sep='\t',
                    item_col=item_col,
                    user_col=user_col if user_col else 'user_id'
                )
                return features
        except Exception as e:
            print(f"Could not create features from interactions: {e}")

    # Fallback: return None (will skip unseen handling)
    print("Warning: No item features available - skipping unseen item handling")
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run RecBLR with unseen item handling.')
    parser.add_argument('--model', type=str, default='R', choices=['B', 'R', 'S'],
                        help='Model to use: B for Bert4Rec, R for RecBLR (default: R), S for SASRec')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['preprocessing', 'postprocessing', 'both'],
                        help='Unseen handling mode: preprocessing, postprocessing, or both')
    parser.add_argument('--n_components', type=int, default=16,
                        help='Number of PCA components for item similarity (default: 16)')
    parser.add_argument('--skip_item_features', action='store_true',
                        help='Skip creating item features (will fail for datasets without features)')
    args = parser.parse_args()

    # Only RecBLR supports unseen handling with our implementation
    if args.model != 'R':
        print(f"Warning: Unseen handling only implemented for RecBLR (model='R')")
        print(f"Running {args.model} without unseen handling")
        model_class = BERT4Rec if args.model == 'B' else SASRec
    else:
        model_class = RecBLRWithUnseen

    config_file = 'config.yaml'
    config = Config(model=model_class, config_file_list=[config_file])

    # Apply specific RecBLR architecture flags only when model is RecBLR
    if args.model != 'R':
        config['bd_lru_only'] = False
        config['disable_conv1d'] = False
        config['disable_ffn'] = False

    init_seed(config['seed'], config['reproducibility'])

    # Create a FileHandler to capture log output
    mode_suffix = f"_{args.mode}" if args.model == 'R' else ""
    log_file_path = f"temp_run_log_{model_class.__name__}{mode_suffix}.log"
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.addHandler(file_handler)

    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = model_class(config, train_data.dataset).to(config['device'])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # ========================================================================
    # Model training (standard, no unseen handling during training)
    # ========================================================================
    logger.info("="*80)
    logger.info("Training phase (standard RecBLR)")
    logger.info("="*80)

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=config["show_progress"]
    )

    # ========================================================================
    # Evaluation with unseen item handling
    # ========================================================================
    if args.model == 'R' and not args.skip_item_features:
        logger.info("\n" + "="*80)
        logger.info(f"Setting up unseen item handling ({args.mode})")
        logger.info("="*80)

        # Get valid items from training vocabulary
        valid_items = dataset.id2token(dataset.iid_field, range(1, dataset.item_num))

        # Create or load item features
        dataset_name = config['dataset']
        item_features = create_item_features_for_dataset(dataset_name, config['data_path'])

        if item_features is not None and len(item_features) > 0:
            # Create and fit unseen item handler
            logger.info(f"Creating UnseenItemHandler with n_components={args.n_components}")

            unseen_handler = UnseenItemHandler(
                item_descriptions=item_features,
                valid_items=list(valid_items),
                n_components=args.n_components,
                random_state=config['seed']
            )
            unseen_handler.fit(verbose=True)

            # Save handler for reuse
            handler_path = f'saved/unseen_handler_{dataset_name}.pkl'
            os.makedirs('saved', exist_ok=True)
            unseen_handler.save(handler_path)
            logger.info(f"Saved UnseenItemHandler to {handler_path}")

            # Enable unseen handling based on mode
            use_preprocessing = args.mode in ['preprocessing', 'both']
            use_postprocessing = args.mode in ['postprocessing', 'both']

            model.enable_unseen_handling(
                unseen_handler,
                use_preprocessing=use_preprocessing,
                use_postprocessing=use_postprocessing,
                verbose=True
            )

            logger.info(f"\nEvaluating with unseen handling mode: {args.mode}")
        else:
            logger.warning("No item features available - skipping unseen handling")
            logger.warning("Evaluation will use standard RecBLR (no preprocessing/postprocessing)")

    # model evaluation
    logger.info("\n" + "="*80)
    logger.info("Evaluation phase")
    logger.info("="*80)

    test_result = trainer.evaluate(
        test_data, show_progress=config["show_progress"]
    )

    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    # Process and plot results
    with open(log_file_path, 'r') as f:
        log_contents = f.read()
    df = parse_log_text(log_contents)

    # Extract filename without extension for output prefix
    output_prefix = f"{model_class.__name__}_{config_file.split('/')[-1].replace('.yaml', '_')}"
    if args.model == 'R':
        output_prefix += f"_{args.mode}"

    generate_plots(df, output_prefix)
    df.to_csv(f"{output_prefix}training_metrics.csv", index=False)
    print(f"Metrics for {config_file} saved to {output_prefix}training_metrics.csv")
    print(f"Plots for {config_file} generated with prefix {output_prefix}")

    # Clean up logger and temporary log file
    logger.removeHandler(file_handler)
    file_handler.close()
    os.remove(log_file_path)
