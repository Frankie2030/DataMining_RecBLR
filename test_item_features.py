"""
Quick test script to debug item features loading
"""
import sys
import pandas as pd
import os

# Copy the function here for testing
def create_item_features_for_dataset(dataset_name, dataset_path='dataset'):
    print(f"Loading item features for {dataset_name}...")

    # Check for pre-created features file
    features_file = os.path.join(dataset_path, dataset_name, f'{dataset_name}_item_features.csv')
    if os.path.exists(features_file):
        print(f"Found pre-created features: {features_file}")
        return pd.read_csv(features_file)

    # Try to load item file
    item_file = os.path.join(dataset_path, dataset_name, f'{dataset_name}.item')
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

    # Fallback: return None (will skip unseen handling)
    print("Warning: No item features available - skipping unseen item handling")
    return None


# Test it
if __name__ == '__main__':
    result = create_item_features_for_dataset('amazon-beauty', 'dataset')
    print(f"\nFinal result: {type(result)}")
    if result is not None:
        print(f"Shape: {result.shape}")
        print(f"Columns: {result.columns.tolist()}")
        print(f"First 3 rows:\n{result.head(3)}")
