"""
Run RecBLR with unseen item handling (preprocessing and/or postprocessing).

This script extends the standard run.py to support evaluation with:
- Preprocessing only: Maps unseen items in input to similar known items
- Postprocessing only: Extends prediction scores to unseen items
- Both: Full preprocessing + postprocessing pipeline

Usage:
    python run_with_unseen.py --model R --mode pre
    python run_with_unseen.py --model R --mode post
    python run_with_unseen.py --model R --mode both
"""

import sys
import logging
from logging import getLogger
import argparse
import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from recbole.data.interaction import Interaction

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
    features_file = os.path.join(dataset_path, f'{dataset_name}_item_features.csv')
    if os.path.exists(features_file):
        print(f"Found pre-created features: {features_file}")
        return pd.read_csv(features_file)

    # Try to load item file
    item_file = os.path.join(dataset_path, f'{dataset_name}.item')
    if os.path.exists(item_file):
        print(f"Found item file: {item_file}")
        try:
            item_df = pd.read_csv(item_file, sep='\t')
            print(f"Loaded {len(item_df)} items from .item file")
            
            # Check column names (RecBole uses :token suffix)
            cols = item_df.columns.tolist()
            print(f"Columns found: {cols}")

            # Find item_id column
            item_id_col = None
            for col in cols:
                if 'item' in col.lower() and 'id' in col.lower():
                    item_id_col = col
                    break

            if item_id_col is None:
                print(f"Could not find item_id column in {item_file}")
            else:
                print(f"Using item_id column: {item_id_col}")
                
                # Use available text columns as description
                # Select columns that are text-based (object dtype or token_seq)
                text_cols = []
                for col in cols:
                    if col != item_id_col:
                        # Include object columns and token_seq columns
                        if ':token' in col or ':token_seq' in col or item_df[col].dtype == 'object':
                            # Skip numeric-looking columns
                            if ':float' not in col and ':int' not in col:
                                text_cols.append(col)
                
                print(f"Text columns for description: {text_cols}")

                if text_cols:
                    print(f"Using item features from {len(text_cols)} columns: {text_cols}")
                    # Create description by concatenating text columns
                    item_df['description'] = item_df[text_cols].apply(
                        lambda x: ' '.join(str(val) for val in x.dropna() if str(val).strip()),
                        axis=1
                    )
                    result = item_df[[item_id_col, 'description']].copy()
                    result.columns = ['item_id', 'description']
                    print(f"Successfully created features for {len(result)} items")
                    print(f"Sample description: {result['description'].iloc[0][:100]}...")
                    return result
                else:
                    print("No suitable text columns found for creating descriptions")
        except Exception as e:
            import traceback
            print(f"ERROR loading item features from {item_file}:")
            print(f"Exception: {e}")
            print(f"Traceback:\n{traceback.format_exc()}")

    # Try to create features from interactions
    inter_file = os.path.join(dataset_path, f'{dataset_name}.inter')
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


def prepare_hm_data(config):
    """
    1. Split users into train/val/test (70/10/20)
    2. Save train+val for RecBole training (we'll use val for monitoring)
    3. Return test split dataframe for manual evaluation
    
    This avoids RecBole's leave-one-out split which never has unseen users/items.
    """
    print("Preparing data with 3-way user split (train/val/test = 70/10/20)...")
    
    # Construct file path
    dataset_name = config['dataset']
    data_path = config['data_path']
    inter_file = os.path.join(data_path, f'{dataset_name}.inter')
    
    if not os.path.exists(inter_file):
        raise FileNotFoundError(f"Interaction file not found: {inter_file}")
    
    # Backup/restore mechanism to prevent double-splitting
    original_backup = inter_file.replace('.inter', '_ORIGINAL.inter')
    
    if os.path.exists(original_backup):
        print(f"Restoring original data from {original_backup}...")
        shutil.copy(original_backup, inter_file)
    else:
        print(f"Backing up original data to {original_backup}...")
        shutil.copy(inter_file, original_backup)
        
    # Read data
    print(f"Reading {inter_file}...")
    df = pd.read_csv(inter_file, sep='\t')
    
    # Identify columns
    ts_col = next((c for c in df.columns if 'timestamp' in c), None)
    user_col = next((c for c in df.columns if 'user' in c), None)
    item_col = next((c for c in df.columns if 'item' in c), None)
    
    if not all([ts_col, user_col, item_col]):
        raise ValueError(f"Could not identify all required columns (user, item, timestamp). Found: {df.columns}")
        
    # Split users into train/val/test (70/10/20)
    print("Splitting users 70/10/20 (train/val/test)...")
    all_users = df.groupby(user_col)[item_col].agg(list).reset_index()[user_col]
    
    # First split: 70% train, 30% temp (val+test)
    train_users, temp_users = train_test_split(all_users, test_size=0.3, random_state=42)
    
    # Second split: split temp into 10% val, 20% test (which is 1/3 and 2/3 of temp)
    val_users, test_users = train_test_split(temp_users, test_size=2/3, random_state=42)
    
    train_df = df[df[user_col].isin(train_users)].copy()
    val_df = df[df[user_col].isin(val_users)].copy()
    test_df = df[df[user_col].isin(test_users)].copy()
    
    print(f"Train set: {len(train_df)} interactions, {train_df[user_col].nunique()} users")
    print(f"Val set:   {len(val_df)} interactions, {val_df[user_col].nunique()} users")
    print(f"Test set:  {len(test_df)} interactions, {test_df[user_col].nunique()} users")
    
    # Save all three splits as separate files for reproducibility
    base_path = inter_file.replace('.inter', '')
    train_file_separate = f"{base_path}_train.inter"
    val_file_separate = f"{base_path}_val.inter"
    test_file_separate = f"{base_path}_test.inter"
    
    print(f"Saving train split to {train_file_separate}...")
    train_df.to_csv(train_file_separate, sep='\t', index=False)
    
    print(f"Saving val split to {val_file_separate}...")
    val_df.to_csv(val_file_separate, sep='\t', index=False)
    
    print(f"Saving test split to {test_file_separate}...")
    test_df.to_csv(test_file_separate, sep='\t', index=False)
    
    # Save train+val for RecBole (it will use leave-one-out on this for internal validation)
    # But we'll do our own evaluation on the held-out test users
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    train_file = inter_file
    print(f"Saving train+val split to {train_file} (for RecBole)...")
    train_val_df.to_csv(train_file, sep='\t', index=False)
    
    
    return test_df, val_df, user_col, item_col, ts_col


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run RecBLR with unseen item handling.')
    parser.add_argument('--model', type=str, default='R', choices=['B', 'R', 'S'],
                        help='Model to use: B for Bert4Rec, R for RecBLR (default: R), S for SASRec')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['none', 'pre', 'post', 'both'],
                        help='Unseen handling mode: none (no handling), pre, post, or both')
    parser.add_argument('--exp', type=str, default=None,
                        help='Experiment type: "unseen" to run all 4 modes')
    parser.add_argument('--n_components', type=int, default=16,
                        help='Number of PCA components for item similarity (default: 16)')
    parser.add_argument('--skip_item_features', action='store_true',
                        help='Skip creating item features (will fail for datasets without features)')
    args = parser.parse_args()
    
    # Check if we should run all modes
    if args.exp == 'unseen' and args.mode is None:
        # Run all 4 modes
        import subprocess
        import sys
        modes = ['none', 'pre', 'post', 'both']
        print(f"Running all modes: {modes}")
        for mode in modes:
            print(f"\n{'='*80}\nRUNNING MODE: {mode}\n{'='*80}\n")
            cmd = [sys.executable, __file__, '--model', args.model, '--mode', mode, 
                   '--n_components', str(args.n_components)]
            if args.skip_item_features:
                cmd.append('--skip_item_features')
            subprocess.run(cmd)
        sys.exit(0)
    
    # Set default mode if not specified
    if args.mode is None:
        args.mode = 'both'

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

    # dataset filtering & splitting (H&M style)
    # We do this BEFORE creating the dataset for RecBole
    # This will overwrite the original .inter file with train split
    test_df, val_df, user_col_name, item_col_name, ts_col_name = prepare_hm_data(config)
    
    # dataset filtering (RecBole standard loading of the TRAIN split)
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting (Internal split for training/validation)
    # We use the train split from H&M as the "full" dataset here, 
    # and let RecBole split it further for training/validation monitoring
    train_data, valid_data, test_data_dummy = data_preparation(config, dataset)

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
    # Setup item similarity for unseen item handling (H&M style)
    # ========================================================================
    item_mapper = None
    valid_mapper = None
    sim_cosine_valid = None
    sim_cosine_all = None
    convert2valid = None
    weighted_similarity = None
    
    if args.model == 'R' and args.mode != 'none' and not args.skip_item_features:
        logger.info("\n" + "="*80)
        logger.info(f"Setting up unseen item handling ({args.mode})")
        logger.info("="*80)

        # Get valid items from training vocabulary
        valid_items = dataset.id2token(dataset.iid_field, range(1, dataset.item_num))
        valid_items_set = set(valid_items)

        # Create or load item features
        dataset_name = config['dataset']
        item_features = create_item_features_for_dataset(dataset_name, config['data_path'])

        if item_features is not None and len(item_features) > 0:
            logger.info(f"Total items in features: {len(item_features)}")
            logger.info(f"Valid items in training: {len(valid_items)}")
            
            # Sort and create mappers (H&M style)
            item_features = item_features.sort_values("item_id").reset_index(drop=True)
            item_mapper = {item_features["item_id"].iloc[i]: i for i in range(len(item_features))}
            item_inv_mapper = {i: item_features["item_id"].iloc[i] for i in range(len(item_features))}
            
            # Create valid item mappers
            valid_ids = [item_mapper[item] for item in valid_items if item in item_mapper]
            valid_mapper = {valid_items[i]: i for i in range(len(valid_items)) if valid_items[i] in item_mapper}
            valid_inv_mapper = {i: valid_items[i] for i in range(len(valid_items)) if valid_items[i] in item_mapper}
            
            logger.info(f"Computing TF-IDF vectors for {len(item_features)} items...")
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            from scipy.sparse import csr_matrix
            from sklearn.metrics.pairwise import cosine_similarity
            from functools import lru_cache
            
            # Compute TF-IDF
            vect = TfidfVectorizer()
            tfidf = vect.fit_transform(item_features["description"])
            X = csr_matrix(tfidf)
            
            # Reduce dimensions
            logger.info(f"Reducing to {args.n_components} components with TruncatedSVD...")
            svd = TruncatedSVD(n_components=args.n_components, n_iter=3, random_state=config['seed'])
            X = svd.fit_transform(X)
            
            # Get valid item vectors
            X_valid = X[valid_ids]
            
            # Compute similarity matrix (all items vs valid items only)
            logger.info(f"Computing cosine similarity matrix ({len(item_features)} x {len(valid_ids)})...")
            sim_cosine_valid = cosine_similarity(X, X_valid)
            
            # For postprocessing: also compute all items vs all items similarity
            if args.mode in ['post', 'both']:
                logger.info(f"Computing full similarity matrix ({len(item_features)} x {len(item_features)}) for postprocessing...")
                sim_cosine_all = cosine_similarity(X, X)
                
                # Setup weighted similarity function (H&M notebook cell 47)
                non_valid_ids = [item_mapper[item] for item in item_mapper.keys() if item not in valid_items_set]
                sim_scores = sim_cosine_valid[non_valid_ids] / np.sum(sim_cosine_valid[non_valid_ids], axis=1).reshape(-1, 1)
                
                def weighted_similarity(initial_scores):
                    """Extend scores from valid items to all items via similarity"""
                    final_scores = np.zeros((X.shape[0]))
                    final_scores[valid_ids] = initial_scores
                    final_scores[non_valid_ids] = np.dot(sim_scores, initial_scores)
                    return final_scores
            
            # Create conversion function (H&M style)
            @lru_cache(maxsize=2048)
            def convert2valid(item):
                """Map an unseen item to the most similar valid item"""
                if item in valid_items_set:
                    return item
                if item not in item_mapper:
                    # Item not in features at all - return a random valid item
                    return list(valid_items)[0]
                item_id = item_mapper[item]
                sim_scores = sim_cosine_valid[item_id]
                best_idx = np.argmax(sim_scores)
                return valid_inv_mapper[best_idx]
            
            logger.info(f"Unseen item handling setup complete!")
            logger.info(f"Mode: {args.mode}")
            if args.mode in ['pre', 'both']:
                logger.info("  - Preprocessing: Will map unseen items to similar valid items")
            if args.mode in ['post', 'both']:
                logger.info("  - Postprocessing: Will extend predictions to all items via similarity")
                
        else:
            logger.warning("No item features available - skipping unseen handling")
            logger.warning("Evaluation will use standard RecBLR (no preprocessing/postprocessing)")
    elif args.mode == 'none':
        logger.info("\n" + "="*80)
        logger.info("Unseen handling mode: NONE")
        logger.info("Using H&M-style 80/20 split WITHOUT unseen item handling")
        logger.info("Unseen items in test data will be filtered out")
        logger.info("="*80)


    # model evaluation
    logger.info("\n" + "="*80)
    logger.info("Evaluation phase (H&M Style)")
    logger.info("="*80)

    # Prepare test sequences from test_df
    print("Preparing test sequences...")
    test_sequence = test_df.sort_values([user_col_name, ts_col_name]) \
                        .groupby(user_col_name)[item_col_name] \
                        .agg(list) \
                        .reset_index()
    test_sequence.columns = ["customer_id", "sequence"]
    
    # Map unseen items if convert2valid function is available (preprocessing mode)
    if convert2valid is not None and args.mode in ['pre', 'both']:
        print("Mapping unseen items in test sequences (preprocessing)...")
        
        # Track how many items were mapped
        mapping_stats = {'original_unseen': 0, 'total': 0, 'mapped': 0}
        valid_items_set_check = set(dataset.id2token(dataset.iid_field, range(1, dataset.item_num)))
        valid_articles = list(valid_items_set_check)  # For notebook compatibility
        
        # H&M-style: map sequence[:-1] to valid items, keep last item as target
        def to_valid_list(item_list):
            """Map unseen items to similar valid items (H&M style - notebook cell 37)"""
            valid_list = []
            # Map all items except the last one (which is the target)
            for i in range(len(item_list) - 1):
                item = item_list[i]
                mapping_stats['total'] += 1
                if item not in valid_articles:
                    mapping_stats['original_unseen'] += 1
                    mapping_stats['mapped'] += 1
                    valid_list.append(convert2valid(item))
                else:
                    valid_list.append(item)
            return ['[PAD]'] if len(valid_list) == 0 else valid_list
        
        test_sequence["item_id_list"] = test_sequence["sequence"].apply(to_valid_list)
        
        # Report mapping statistics
        if mapping_stats['original_unseen'] > 0:
            print(f"Preprocessing: Mapped {mapping_stats['original_unseen']}/{mapping_stats['total']} " +
                  f"({100*mapping_stats['original_unseen']/mapping_stats['total']:.2f}%) unseen items to similar valid items")
        else:
            print(f"Preprocessing: All {mapping_stats['total']} items were already in training vocabulary")
    else:
        print("No preprocessing - using raw sequences")
        # H&M style: use all items except last as input
        test_sequence["item_id_list"] = test_sequence["sequence"].apply(
            lambda seq: seq[:-1] if len(seq) > 1 else ['[PAD]']
        )
    
    test_sequence["item_length"] = test_sequence["item_id_list"].apply(len)
    
    # Evaluation Loop - Notebook style (cells 44 and 48)
    print(f"Evaluating on {len(test_sequence)} users...")
    model.eval()
    
    from sklearn.metrics import ndcg_score
    
    # Determine output shape based on mode
    if args.mode in ['post', 'both'] and weighted_similarity is not None:
        # Postprocessing: evaluate on ALL items (including unseen)
        output_shape = (len(test_sequence), len(item_mapper))
        print(f"Postprocessing mode: Evaluating on {len(item_mapper)} total items (including unseen)")
    else:
        # No postprocessing: evaluate only on valid items
        output_shape = (len(test_sequence), dataset.item_num - 1)
        print(f"No postprocessing: Evaluating on {dataset.item_num - 1} valid items only")
    
    y_scores = np.zeros(output_shape)
    y_true = np.zeros(output_shape)
    
    device = config['device']
    valid_eval_count = 0
    
    with torch.no_grad():
        for i, row in test_sequence.iterrows():
            if i % 1000 == 0:
                print(f"Processed {i}/{len(test_sequence)} users")
            
            # Convert item tokens to IDs (notebook cell 44)
            try:
                item_id_list = np.array([dataset.token2id(dataset.iid_field, row["item_id_list"])])
            except (ValueError, KeyError):
                # Skip if any item in sequence is not in vocabulary
                continue
            
            interaction = {
                "item_id_list": torch.LongTensor(item_id_list).to(device),
                "item_length": torch.LongTensor(np.array([row["item_length"]])).to(device)
            }
            
            scores = model.full_sort_predict(interaction)[0]
            scores = scores.cpu().detach().numpy()
            
            # Get true target item
            true_item = row["sequence"][-1]
            
            # Apply postprocessing if enabled
            if args.mode in ['post', 'both'] and weighted_similarity is not None:
                # Notebook cell 48: Extend scores to all items
                y_scores[i] = weighted_similarity(scores[1:])  # Skip padding item
                
                # Set ground truth
                if true_item in item_mapper:
                    true_item_id = item_mapper[true_item]
                    y_true[i, true_item_id] = 1
                    valid_eval_count += 1
                else:
                    # True item not in features at all - skip
                    continue
            else:
                # No postprocessing: use scores only for valid items
                y_scores[i] = scores[1:]  # Skip padding item
                
                # Set ground truth - only if target is in training vocabulary
                try:
                    # If preprocessing was used, target might have been mapped
                    if args.mode in ['pre', 'both'] and convert2valid is not None:
                        if true_item not in valid_articles:
                            true_item = convert2valid(true_item)
                    
                    true_item_id = dataset.token2id(dataset.iid_field, true_item) - 1
                    y_true[i, true_item_id] = 1
                    valid_eval_count += 1
                except (ValueError, KeyError):
                    # Target item is unseen and no preprocessing - skip this user
                    continue
    
    # Compute metrics using sklearn (notebook style)
    print(f"Computing metrics for {valid_eval_count} valid evaluations...")
    
    # Filter out rows where no valid target was found
    valid_rows = y_true.sum(axis=1) > 0
    y_true_filtered = y_true[valid_rows]
    y_scores_filtered = y_scores[valid_rows]
    
    if len(y_true_filtered) > 0:
        ndcg_10 = ndcg_score(y_true_filtered, y_scores_filtered, k=10)
        
        # Compute Hit@10 manually
        hit_10_scores = []
        for i in range(len(y_true_filtered)):
            topk_indices = np.argpartition(y_scores_filtered[i], -10)[-10:]
            true_indices = np.where(y_true_filtered[i] == 1)[0]
            hit = 1 if len(np.intersect1d(topk_indices, true_indices)) > 0 else 0
            hit_10_scores.append(hit)
        hit_10 = np.mean(hit_10_scores)
    else:
        ndcg_10 = 0.0
        hit_10 = 0.0
        print("WARNING: No valid evaluations found!")
    
    test_result = {
        'hit@10': hit_10,
        'ndcg@10': ndcg_10
    }

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
