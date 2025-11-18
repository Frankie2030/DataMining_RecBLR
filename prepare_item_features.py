"""
Utility to prepare item features for unseen item handling.

For datasets without rich item metadata, this creates synthetic features
based on interaction patterns and item statistics.
"""

import pandas as pd
import numpy as np
import os
from collections import Counter


def create_interaction_based_features(
    interaction_file,
    output_file,
    sep='\t',
    item_col='item_id',
    user_col='user_id',
    timestamp_col='timestamp'
):
    """
    Create item features based on interaction patterns.

    For datasets without item metadata, we can create features from:
    - Item interaction frequency
    - User diversity (how many different users interact)
    - Temporal patterns
    - Co-occurrence with other items

    Args:
        interaction_file: Path to interaction file
        output_file: Path to save item features
        sep: Separator in interaction file
        item_col: Name of item column
        user_col: Name of user column
        timestamp_col: Name of timestamp column (optional)

    Returns:
        DataFrame with item_id and description columns
    """
    print(f"Loading interactions from {interaction_file}...")
    df = pd.read_csv(interaction_file, sep=sep)

    # Ensure columns exist
    if item_col not in df.columns:
        raise ValueError(f"Column {item_col} not found in interaction file")

    print(f"Processing {len(df)} interactions for {df[item_col].nunique()} unique items...")

    # Compute item statistics
    item_stats = []

    for item_id in df[item_col].unique():
        item_data = df[df[item_col] == item_id]

        # Basic statistics
        num_interactions = len(item_data)
        num_users = item_data[user_col].nunique() if user_col in df.columns else 0

        # Create description from statistics
        desc_parts = [
            f"item_{item_id}",
            f"interactions_{num_interactions}",
            f"users_{num_users}",
        ]

        # Add frequency bin
        if num_interactions < 10:
            desc_parts.append("rare_item")
        elif num_interactions < 100:
            desc_parts.append("medium_item")
        else:
            desc_parts.append("popular_item")

        # Add user diversity bin
        if num_users > 0:
            diversity = num_users / num_interactions
            if diversity > 0.8:
                desc_parts.append("high_diversity")
            elif diversity > 0.5:
                desc_parts.append("medium_diversity")
            else:
                desc_parts.append("low_diversity")

        description = " ".join(desc_parts)

        item_stats.append({
            'item_id': str(item_id),
            'description': description
        })

    # Create DataFrame
    item_features = pd.DataFrame(item_stats)

    # Save to file
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    item_features.to_csv(output_file, index=False)

    print(f"✓ Saved item features for {len(item_features)} items to {output_file}")

    return item_features


def prepare_features_for_dataset(dataset_name, dataset_path='dataset'):
    """
    Prepare item features for a RecBole dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'ml-1m', 'yelp')
        dataset_path: Path to dataset directory

    Returns:
        Path to created item features file
    """
    # Paths
    dataset_dir = os.path.join(dataset_path, dataset_name)
    inter_file = os.path.join(dataset_dir, f'{dataset_name}.inter')
    item_file = os.path.join(dataset_dir, f'{dataset_name}.item')
    features_file = os.path.join(dataset_dir, f'{dataset_name}_item_features.csv')

    # Check if item file already exists with good features
    if os.path.exists(item_file):
        try:
            item_df = pd.read_csv(item_file, sep='\t')
            # Check if we have text columns beyond just item_id
            text_cols = item_df.select_dtypes(include=['object']).columns.tolist()
            if 'item_id' in text_cols:
                text_cols.remove('item_id')

            if len(text_cols) > 0:
                print(f"Using existing item file with features: {item_file}")
                # Create description from existing features
                item_df['description'] = item_df[text_cols].apply(
                    lambda x: ' '.join(x.dropna().astype(str)),
                    axis=1
                )
                item_df = item_df[['item_id:token', 'description']]
                item_df.columns = ['item_id', 'description']
                item_df.to_csv(features_file, index=False)
                return features_file
        except Exception as e:
            print(f"Could not use existing item file: {e}")

    # Create features from interactions
    if os.path.exists(inter_file):
        print(f"Creating features from interaction file: {inter_file}")

        # Determine column names (RecBole uses :token suffix)
        sample_df = pd.read_csv(inter_file, sep='\t', nrows=5)
        cols = sample_df.columns.tolist()

        # Find item and user columns
        item_col = None
        user_col = None
        for col in cols:
            if 'item' in col.lower():
                item_col = col
            if 'user' in col.lower():
                user_col = col

        if item_col is None:
            raise ValueError(f"Could not find item column in {inter_file}")

        # Create features
        create_interaction_based_features(
            inter_file,
            features_file,
            sep='\t',
            item_col=item_col,
            user_col=user_col if user_col else 'user_id'
        )

        return features_file
    else:
        raise FileNotFoundError(f"Could not find interaction file: {inter_file}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Prepare item features for datasets')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., ml-1m, yelp)')
    parser.add_argument('--dataset_path', type=str, default='dataset',
                        help='Path to dataset directory')
    args = parser.parse_args()

    try:
        features_file = prepare_features_for_dataset(args.dataset, args.dataset_path)
        print(f"\n✓ Success! Item features saved to: {features_file}")
        print(f"\nYou can now use these features with run_with_unseen.py")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
