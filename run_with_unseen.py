"""
Run RecBLR with unseen item handling (preprocessing).

This script extends run.py to handle unseen items in test data using the 
preprocessing approach from hmkaggle.ipynb.

Usage:
    python run_with_unseen.py --model R --mode pre
    python run_with_unseen.py --model R --mode none
"""

import sys
import logging
from logging import getLogger
import argparse
import os
import pandas as pd
import numpy as np
import torch
from functools import lru_cache
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import ndcg_score
from scipy.sparse import csr_matrix

from recbole.utils import init_logger, init_seed, set_color, get_flops, get_environment
from recbole.trainer import Trainer
from RecBLR import RecBLR
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from plot_utils import parse_log_text, generate_plots


def prepare_data_split(config):
    """
    1. Filter to last 1/8 of data by timestamp
    2. Split data into train/test by users (80/20 split)
    3. Returns test_df for manual evaluation
    Does NOT overwrite original files.
    """
    print("Preparing data split...")
    
    dataset_name = config['dataset']
    data_path = config['data_path']
    inter_file = os.path.join(data_path, f'{dataset_name}.inter')
    train_file = os.path.join(data_path, f'{dataset_name}_train.inter')
    test_file = os.path.join(data_path, f'{dataset_name}_test.inter')
    
    # Check if split already exists
    if os.path.exists(train_file) and os.path.exists(test_file):
        print(f"Using existing split: {train_file} and {test_file}")
        test_df = pd.read_csv(test_file, sep='\t')
        
        # Get column names
        user_col = next((c for c in test_df.columns if 'user' in c.lower()), None)
        item_col = next((c for c in test_df.columns if 'item' in c.lower()), None)
        ts_col = next((c for c in test_df.columns if 'timestamp' in c.lower()), None)
        
        return test_df, user_col, item_col, ts_col
    
    # Read original data
    print(f"Reading original data from {inter_file}...")
    df = pd.read_csv(inter_file, sep='\t')
    
    # Identify columns
    user_col = next((c for c in df.columns if 'user' in c.lower()), None)
    item_col = next((c for c in df.columns if 'item' in c.lower()), None)
    ts_col = next((c for c in df.columns if 'timestamp' in c.lower()), None)
    
    # Filter to last 1/8 of data by timestamp (like notebook but 1/8 instead of 1/64)
    print(f"Original data: {len(df)} interactions")
    df = df.sort_values(by=ts_col)
    size = len(df)
    df = df.iloc[-size//8:]  # Take last 1/8
    print(f"After filtering to last 1/8: {len(df)} interactions")
    
    # Split users 80/20
    all_users = df[user_col].unique()
    train_users, test_users = train_test_split(all_users, test_size=0.2, random_state=42)
    
    train_df = df[df[user_col].isin(train_users)]
    test_df = df[df[user_col].isin(test_users)]
    
    print(f"Train: {len(train_df)} interactions, {len(train_users)} users")
    print(f"Test: {len(test_df)} interactions, {len(test_users)} users")
    
    # Save splits to separate files (DO NOT overwrite original)
    print(f"Saving train split to {train_file}...")
    train_df.to_csv(train_file, sep='\t', index=False)
    
    print(f"Saving test split to {test_file}...")
    test_df.to_csv(test_file, sep='\t', index=False)
    
    print(f"Original file {inter_file} preserved!")
    
    return test_df, user_col, item_col, ts_col


def load_item_features(dataset_name, data_path):
    """Load item features for similarity computation."""
    print(f"Loading item features for {dataset_name}...")
    
    # Try to load .item file
    item_file = os.path.join(data_path, f'{dataset_name}.item')
    if os.path.exists(item_file):
        item_df = pd.read_csv(item_file, sep='\t')
        
        # Find item_id column
        item_id_col = next((c for c in item_df.columns if 'item' in c.lower() and 'id' in c.lower()), None)
        
        # Get text columns
        text_cols = [c for c in item_df.columns 
                     if c != item_id_col and ':token' in c or item_df[c].dtype == 'object']
        
        if text_cols:
            # Create description from text columns
            item_df['description'] = item_df[text_cols].apply(
                lambda x: ' '.join(str(val) for val in x.dropna() if str(val).strip()),
                axis=1
            )
            result = item_df[[item_id_col, 'description']].copy()
            result.columns = ['item_id', 'description']
            print(f"Loaded {len(result)} items with features")
            return result
    
    print("No item features found")
    return None


def setup_item_similarity(item_features, valid_items, n_components=16, seed=42):
    """
    Setup item similarity computation (notebook cells 19-23).
    Returns mappers and similarity matrix.
    """
    print("Setting up item similarity...")
    
    # Sort and create mappers
    item_features = item_features.sort_values("item_id").reset_index(drop=True)
    item_mapper = {item_features["item_id"].iloc[i]: i for i in range(len(item_features))}
    
    # Create valid item mappers
    valid_ids = [item_mapper[item] for item in valid_items if item in item_mapper]
    valid_inv_mapper = {i: valid_items[i] for i in range(len(valid_items)) if valid_items[i] in item_mapper}
    
    print(f"Computing TF-IDF for {len(item_features)} items...")
    # TF-IDF transform
    vect = TfidfVectorizer()
    tfidf = vect.fit_transform(item_features["description"])
    X = csr_matrix(tfidf)
    
    # Dimensionality reduction
    print(f"Reducing to {n_components} components...")
    svd = TruncatedSVD(n_components=n_components, n_iter=3, random_state=seed)
    X = svd.fit_transform(X)
    
    # Get valid item vectors
    X_valid = X[valid_ids]
    
    # Compute similarity matrix
    print(f"Computing similarity matrix ({len(item_features)} x {len(valid_ids)})...")
    sim_cosine_valid = cosine_similarity(X, X_valid)
    
    return item_mapper, valid_inv_mapper, sim_cosine_valid


def create_preprocessing_function(item_mapper, valid_inv_mapper, sim_cosine_valid, valid_articles):
    """
    Create the to_valid_list function (notebook cell 37).
    """
    def to_valid_list(item_list):
        """Map unseen items to similar valid items."""
        
        @lru_cache(maxsize=2048)  
        def convert2valid(item):
            item_id = item_mapper[item]
            sim_scores = sim_cosine_valid[item_id]
            sort_indices = np.argsort(-sim_scores)
            return valid_inv_mapper[sort_indices[0]]

        valid_list = []
        for i in range(len(item_list)-1):
            if item_list[i] not in valid_articles:
                valid_list.append(convert2valid(item_list[i]))
            else:
                valid_list.append(item_list[i])
                
        return ['[PAD]'] if len(valid_list) == 0 else valid_list
    
    return to_valid_list


def evaluate_with_preprocessing(model, test_sequence, dataset, device, mode='pre'):
    """
    Evaluate model on test sequences (notebook cell 44).
    """
    print(f"Evaluating {len(test_sequence)} users...")
    model.eval()
    
    output_shape = (len(test_sequence), dataset.item_num - 1)
    y_scores = np.zeros(output_shape)
    y_true = np.zeros(output_shape)
    
    valid_eval_count = 0
    
    with torch.no_grad():
        for i, row in test_sequence.iterrows():
            # Convert item tokens to IDs
            try:
                item_id_list = np.array([dataset.token2id(dataset.iid_field, row["item_id_list"])])
            except (ValueError, KeyError):
                continue
            
            interaction = {
                "item_id_list": torch.LongTensor(item_id_list).to(device),
                "item_length": torch.LongTensor(np.array([row["item_length"]])).to(device)
            }
            
            scores = model.full_sort_predict(interaction)[0]
            scores = scores.cpu().detach().numpy()
            
            # Get true target item
            true_item = row["sequence"][-1]
            
            # Scores for valid items only
            y_scores[i] = scores[1:]  # Skip padding
            
            # Set ground truth
            try:
                true_item_id = dataset.token2id(dataset.iid_field, true_item) - 1
                y_true[i, true_item_id] = 1
                valid_eval_count += 1
            except (ValueError, KeyError):
                continue
    
    # Compute metrics
    print(f"Computing metrics for {valid_eval_count} valid evaluations...")
    valid_rows = y_true.sum(axis=1) > 0
    y_true_filtered = y_true[valid_rows]
    y_scores_filtered = y_scores[valid_rows]
    
    if len(y_true_filtered) > 0:
        ndcg_10 = ndcg_score(y_true_filtered, y_scores_filtered, k=10)
        
        # Compute Hit@10
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
    
    return {'hit@10': hit_10, 'ndcg@10': ndcg_10}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run RecBLR with unseen item handling.')
    parser.add_argument('--mode', type=str, default='pre', choices=['none', 'pre'],
                        help='Unseen handling: none or pre (preprocessing)')
    parser.add_argument('--n_components', type=int, default=16,
                        help='PCA components for similarity (default: 16)')
    args = parser.parse_args()
    
    # Model is always RecBLR
    model_class = RecBLR
    
    config_file = 'config.yaml'
    config = Config(model=model_class, config_file_list=[config_file])
    
    init_seed(config['seed'], config['reproducibility'])
    
    # Setup logging
    log_file_path = f"temp_run_log_RecBLR_{args.mode}.log"
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    init_logger(config)
    logger = getLogger()
    logger.addHandler(file_handler)
    
    logger.info(sys.argv)
    logger.info(config)
    
    # ========================================================================
    # Data preparation (notebook cells 1-13)
    # ========================================================================
    test_df, user_col, item_col, ts_col = prepare_data_split(config)
    
    # Point RecBole to the train split
    train_file = os.path.join(config['data_path'], f"{config['dataset']}_train.inter")
    if os.path.exists(train_file):
        # Temporarily rename files so RecBole loads the train split
        original_file = os.path.join(config['data_path'], f"{config['dataset']}.inter")
        backup_file = os.path.join(config['data_path'], f"{config['dataset']}_original.inter")
        
        # Backup original if not already backed up
        if not os.path.exists(backup_file):
            os.rename(original_file, backup_file)
        
        # Copy train split to main file for RecBole
        import shutil
        shutil.copy(train_file, original_file)
        print(f"RecBole will use train split: {train_file}")
    
    # ========================================================================
    # Model training (notebook cells 14-15)
    # ========================================================================
    dataset = create_dataset(config)
    logger.info(dataset)
    
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = model_class(config, train_data.dataset).to(config['device'])
    logger.info(model)
    
    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")
    
    trainer = Trainer(config, model)
    
    logger.info("="*80)
    logger.info("Training phase")
    logger.info("="*80)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=config["show_progress"]
    )
    
    # ========================================================================
    # Item similarity setup (notebook cells 16-23)
    # ========================================================================
    item_mapper = None
    valid_inv_mapper = None
    sim_cosine_valid = None
    
    if args.mode == 'pre':
        logger.info("="*80)
        logger.info("Setting up item similarity for preprocessing")
        logger.info("="*80)
        
        valid_items = dataset.id2token(dataset.iid_field, range(1, dataset.item_num))
        item_features = load_item_features(config['dataset'], config['data_path'])
        
        if item_features is not None:
            item_mapper, valid_inv_mapper, sim_cosine_valid = setup_item_similarity(
                item_features, valid_items, args.n_components, config['seed']
            )
            logger.info("Item similarity setup complete!")
        else:
            logger.warning("No item features - falling back to mode='none'")
            args.mode = 'none'
    
    # ========================================================================
    # Evaluation (notebook cells 24-28)
    # ========================================================================
    logger.info("="*80)
    logger.info(f"Evaluation phase (mode={args.mode})")
    logger.info("="*80)
    
    # Prepare test sequences
    print("Preparing test sequences...")
    test_sequence = test_df.sort_values([user_col, ts_col]) \
                        .groupby(user_col)[item_col] \
                        .agg(list) \
                        .reset_index()
    test_sequence.columns = ["customer_id", "sequence"]
    
    # Get valid articles
    valid_articles = dataset.id2token(dataset.iid_field, range(1, dataset.item_num))
    
    # Apply preprocessing if enabled
    if args.mode == 'pre' and item_mapper is not None:
        print("Applying preprocessing...")
        to_valid_list = create_preprocessing_function(
            item_mapper, valid_inv_mapper, sim_cosine_valid, valid_articles
        )
        test_sequence["item_id_list"] = test_sequence["sequence"].apply(to_valid_list)
    else:
        print("No preprocessing - using raw sequences")
        test_sequence["item_id_list"] = test_sequence["sequence"].apply(
            lambda seq: seq[:-1] if len(seq) > 1 else ['[PAD]']
        )
    
    test_sequence["item_length"] = test_sequence["item_id_list"].apply(len)
    
    # Evaluate
    test_result = evaluate_with_preprocessing(
        model, test_sequence, dataset, config['device'], args.mode
    )
    
    # ========================================================================
    # Results
    # ========================================================================
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
    
    output_prefix = f"RecBLR_{config_file.split('/')[-1].replace('.yaml', '')}_{args.mode}"
    generate_plots(df, output_prefix)
    df.to_csv(f"{output_prefix}_training_metrics.csv", index=False)
    print(f"Metrics saved to {output_prefix}_training_metrics.csv")
    print(f"Plots generated with prefix {output_prefix}")
    
    # Clean up
    logger.removeHandler(file_handler)
    file_handler.close()
    os.remove(log_file_path)
