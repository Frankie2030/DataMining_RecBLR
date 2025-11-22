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


def prepare_hm_data(config):
    """
    1. Split users 80/20 (Train/Test)
    2. Save train split for RecBole training
    3. Return test split dataframe for manual evaluation
    """
    print("Preparing data with H&M-style splitting...")
    
    # Construct file path
    dataset_name = config['dataset']
    data_path = config['data_path']
    inter_file = os.path.join(data_path, f'{dataset_name}.inter')
    
    if not os.path.exists(inter_file):
        raise FileNotFoundError(f"Interaction file not found: {inter_file}")
        
    # Read data
    print(f"Reading {inter_file}...")
    df = pd.read_csv(inter_file, sep='\t')
    
    # Identify columns
    ts_col = next((c for c in df.columns if 'timestamp' in c), None)
    user_col = next((c for c in df.columns if 'user' in c), None)
    item_col = next((c for c in df.columns if 'item' in c), None)
    
    if not all([ts_col, user_col, item_col]):
        raise ValueError(f"Could not identify all required columns (user, item, timestamp). Found: {df.columns}")
        
    # # 1. Filter by timestamp (last 1/64th)
    # print("Filtering by timestamp (last 1/64th)...")
    # # Ensure timestamp is numeric
    # # df[ts_col] = pd.to_numeric(df[ts_col], errors='coerce') 
    # # Assuming it's already float/int based on RecBole format, but let's be safe
    
    # quantile_val = df[ts_col].quantile(1 - 1/64)
    # data = df[df[ts_col] > quantile_val].copy()
    # print(f"Filtered data size: {len(data)} (Original: {len(df)})")
    
    # 2. Split users
    print("Splitting users 80/20...")
    user_seqs = df.groupby(user_col)[item_col].agg(list).reset_index()[user_col]
    train_ids, test_ids = train_test_split(user_seqs, test_size=0.2, random_state=42)
    
    train_df = df[df[user_col].isin(train_ids)].copy()
    test_df = df[df[user_col].isin(test_ids)].copy()
    
    print(f"Train set: {len(train_df)} interactions, {train_df[user_col].nunique()} users")
    print(f"Test set: {len(test_df)} interactions, {test_df[user_col].nunique()} users")
    
    # 3. Save train split to original dataset directory
    # Overwrite the original .inter file with train split
    train_file = inter_file
    print(f"Saving train split to {train_file}...")
    train_df.to_csv(train_file, sep='\t', index=False)
    
    return test_df, user_col, item_col, ts_col


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

    # dataset filtering & splitting (H&M style)
    # We do this BEFORE creating the dataset for RecBole
    # This will overwrite the original .inter file with train split
    test_df, user_col_name, item_col_name, ts_col_name = prepare_hm_data(config)
    
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
            # Add method to map list of tokens
            def map_list_to_valid(self, item_list):
                # Exclude last item if we are following H&M logic? 
                # No, map everything, we slice later.
                valid_list = []
                # H&M logic: iterate range(len(item_list)-1) -> they DROP the last item from the input list
                # But we want to map the WHOLE list so we can extract target later.
                # Wait, H&M `to_valid_list` returns a list of length N-1.
                # So `test_sequence['item_id_list']` has length N-1.
                # And `test_sequence['sequence']` has length N.
                # So we should probably implement `map_list_to_valid` to return the full mapped list,
                # and then slice it in the eval loop.
                
                for item in item_list:
                    if item in self.valid_items_set:
                        valid_list.append(item)
                    else:
                        valid_list.append(self.get_most_similar_item(item))
                return valid_list
            
            # Monkey patch the handler or just use the logic here
            unseen_handler.map_list_to_valid = map_list_to_valid.__get__(unseen_handler)
            
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

    # model evaluation
    logger.info("\n" + "="*80)
    logger.info("Evaluation phase (Custom H&M Style)")
    logger.info("="*80)

    # Prepare test sequences from test_df
    print("Preparing test sequences...")
    test_sequence = test_df.sort_values([user_col_name, ts_col_name]) \
                        .groupby(user_col_name)[item_col_name] \
                        .agg(list) \
                        .reset_index()
    test_sequence.columns = ["customer_id", "sequence"]
    
    # Map unseen items if handler is available
    if 'unseen_handler' in locals() and unseen_handler is not None:
        print("Mapping unseen items in test sequences...")
        # We use the handler's internal mapping logic
        # Note: unseen_handler.map_to_valid expects a list of tokens
        # We need to apply it to each sequence
        
        # Define a helper to map a single sequence
        def map_seq(seq):
            return unseen_handler.map_list_to_valid(seq)
            
        test_sequence["item_id_list_tokens"] = test_sequence["sequence"].apply(map_seq)
    else:
        print("No unseen handler - using raw sequences (unseen items might be dropped/padded)")
        test_sequence["item_id_list_tokens"] = test_sequence["sequence"]

    # Convert tokens to IDs
    print("Converting tokens to IDs...")
    def tokens_to_ids(tokens):
        # RecBole's token2id returns 0 for unknown tokens
        # We assume tokens are already mapped to valid ones if handler was used
        return [dataset.token2id(dataset.iid_field, t) for t in tokens]

    test_sequence["item_id_list"] = test_sequence["item_id_list_tokens"].apply(tokens_to_ids)
    test_sequence["item_length"] = test_sequence["item_id_list"].apply(len)
    
    # Evaluation Loop
    print(f"Evaluating on {len(test_sequence)} users...")
    model.eval()
    
    # Metrics
    from recbole.evaluator.metrics import Hit, NDCG, MRR
    # We can use RecBole metrics or sklearn. Let's use sklearn to match H&M exactly if possible,
    # or just implement simple Hit/NDCG calculation.
    # H&M notebook uses ndcg_score from sklearn.
    from sklearn.metrics import ndcg_score
    
    ndcg_10_scores = []
    hit_10_scores = []
    
    device = config['device']
    
    with torch.no_grad():
        for i, row in test_sequence.iterrows():
            if i % 1000 == 0:
                print(f"Processed {i}/{len(test_sequence)} users")
                
            seq_ids = row["item_id_list"]
            if len(seq_ids) == 0:
                continue
                
            # Input: sequence excluding the last item (leave-one-out for prediction)
            # OR H&M style: use full sequence to predict NEXT?
            # H&M notebook: 
            # true_item = row["sequence"][-1]
            # item_id_list = ... (it seems they use the full sequence?)
            # Let's check H&M notebook again.
            # "item_id_list" in H&M notebook comes from "sequence".
            # "sequence" is the list of ALL interactions for that user in the test set.
            # Wait, test_df contains interactions.
            # If a user has 5 interactions in test_df, do we predict the 6th?
            # Or do we use 4 to predict 5th?
            # H&M notebook:
            # true_item = row["sequence"][-1]
            # item_id_list = ... row["item_id_list"] ...
            # It seems they use the WHOLE list as input?
            # No, RecBole models usually take history.
            # If they pass the whole list, the model might predict the *next* after the last.
            # BUT `true_item` is the LAST item of the sequence.
            # So they must be using `sequence[:-1]` as input?
            # Let's look at H&M notebook cell 44 again carefully.
            # item_id_list = np.array([dataset.token2id(dataset.iid_field, row["item_id_list"])])
            # It uses the FULL list from the dataframe.
            # AND `true_item = row["sequence"][-1]`.
            # This implies the "sequence" column includes the target.
            # If they pass the full list to the model, RecBole's `forward` typically processes the whole sequence.
            # `full_sort_predict` typically uses the *last* state of the sequence to predict the *next* item.
            # If the sequence includes the target, then we are predicting the item *after* the target.
            # UNLESS `row["item_id_list"]` was constructed to EXCLUDE the last item?
            # In cell 37: `to_valid_list(item_list)` iterates `range(len(item_list)-1)`.
            # AHA! `range(len(item_list)-1)` excludes the last item!
            # So `item_id_list` in H&M notebook IS `sequence[:-1]`.
            
            # So we must do the same: Input is sequence[:-1], Target is sequence[-1].
            
            input_seq = seq_ids[:-1]
            target_item = seq_ids[-1] # This is the ID of the target (mapped or raw?)
            # H&M notebook uses `true_item = row["sequence"][-1]` (RAW token) for ground truth.
            # And maps it to ID for metric calculation.
            
            if len(input_seq) == 0:
                continue
                
            interaction = {
                "item_id_list": torch.LongTensor([input_seq]).to(device),
                "item_length": torch.LongTensor([len(input_seq)]).to(device)
            }
            
            scores = model.full_sort_predict(interaction)[0] # [n_items]
            scores = scores.cpu().detach().numpy()
            
            # Mask the padding item (0)
            scores[0] = -np.inf
            
            # Get top K
            k = 10
            topk_indices = np.argpartition(scores, -k)[-k:]
            topk_indices = topk_indices[np.argsort(scores[topk_indices])[::-1]]
            
            # Calculate Ground Truth ID
            # We need the ID of the true item.
            # If the true item was unseen, it might be mapped to a valid ID in our `seq_ids`?
            # No, `seq_ids` comes from `item_id_list_tokens` which was mapped.
            # So `target_item` is the ID of the mapped true item.
            # H&M notebook: `true_item = convert2valid(true_item) if ... else true_item`
            # So yes, we evaluate against the MAPPED target if it was unseen.
            
            target_id = target_item
            
            # Hit@10
            hit = 1 if target_id in topk_indices else 0
            hit_10_scores.append(hit)
            
            # NDCG@10
            if hit:
                rank = np.where(topk_indices == target_id)[0][0]
                ndcg = 1.0 / np.log2(rank + 2)
            else:
                ndcg = 0
            ndcg_10_scores.append(ndcg)

    avg_hit = np.mean(hit_10_scores)
    avg_ndcg = np.mean(ndcg_10_scores)
    
    test_result = {
        'hit@10': avg_hit,
        'ndcg@10': avg_ndcg
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
