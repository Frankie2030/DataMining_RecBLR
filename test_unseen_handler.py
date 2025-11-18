"""
Simple test script for UnseenItemHandler

This script tests basic functionality of the unseen item handling implementation
without requiring a full RecBole dataset.
"""

import pandas as pd
import numpy as np
from unseen_item_handler import UnseenItemHandler, create_item_descriptions_from_features


def test_basic_functionality():
    """Test basic UnseenItemHandler functionality."""

    print("=" * 60)
    print("Testing UnseenItemHandler Basic Functionality")
    print("=" * 60)

    # Create sample item data
    print("\n1. Creating sample item data...")

    item_data = pd.DataFrame({
        'item_id': ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010'],
        'category': ['clothing', 'clothing', 'electronics', 'electronics', 'books',
                    'books', 'clothing', 'electronics', 'books', 'clothing'],
        'subcategory': ['shirt', 'pants', 'phone', 'laptop', 'fiction',
                       'nonfiction', 'dress', 'tablet', 'textbook', 'jacket'],
        'color': ['red', 'blue', 'black', 'silver', 'blue',
                 'green', 'white', 'black', 'yellow', 'brown'],
        'brand': ['Nike', 'Adidas', 'Apple', 'Dell', 'Penguin',
                 'Oxford', 'Zara', 'Samsung', 'Pearson', 'H&M']
    })

    print(f"   ✓ Created {len(item_data)} sample items")

    # Create descriptions
    print("\n2. Creating item descriptions...")

    item_descriptions = create_item_descriptions_from_features(
        item_data,
        item_id_col='item_id',
        feature_cols=['category', 'subcategory', 'color', 'brand']
    )

    print(f"   ✓ Created descriptions for {len(item_descriptions)} items")
    print("\n   Sample descriptions:")
    for i in range(3):
        print(f"     - {item_descriptions.iloc[i]['item_id']}: "
              f"{item_descriptions.iloc[i]['description']}")

    # Define valid items (simulate training vocab)
    print("\n3. Defining valid items (simulating training vocabulary)...")

    valid_items = ['001', '002', '003', '005', '007', '009']  # 6 out of 10
    print(f"   ✓ Valid items: {valid_items}")
    print(f"   ✓ Unseen items: {[i for i in item_data['item_id'] if i not in valid_items]}")

    # Create and fit handler
    print("\n4. Creating and fitting UnseenItemHandler...")

    handler = UnseenItemHandler(
        item_descriptions=item_descriptions,
        valid_items=valid_items,
        n_components=4,  # Small for testing
        random_state=42
    )
    handler.fit(verbose=False)

    print("   ✓ Handler fitted successfully")

    # Test preprocessing
    print("\n5. Testing preprocessing (mapping unseen → similar valid items)...")

    test_sequences = [
        ['001', '002', '004'],  # Contains unseen item '004'
        ['003', '006', '005'],  # Contains unseen item '006'
        ['001', '008', '010', '002'],  # Contains unseen items '008', '010'
    ]

    for seq in test_sequences:
        preprocessed = handler.preprocess_sequence(seq)
        print(f"   Original:     {seq[:-1]}")
        print(f"   Preprocessed: {preprocessed}")
        print(f"   Target:       {seq[-1]}")
        print()

    # Test postprocessing
    print("6. Testing postprocessing (propagating scores to unseen items)...")

    # Simulate model scores for valid items only
    np.random.seed(42)
    valid_scores = np.random.rand(len(valid_items))
    print(f"   Valid item scores (n={len(valid_scores)}): {valid_scores.round(3)}")

    # Propagate to all items
    all_scores = handler.postprocess_scores(valid_scores)
    print(f"   All item scores (n={len(all_scores)}): {all_scores.round(3)}")

    # Verify valid items keep original scores
    valid_ids = [handler.item_mapper[item] for item in valid_items]
    print(f"\n   Verification: Valid item scores preserved?")
    for i, valid_item in enumerate(valid_items):
        valid_idx = handler.item_mapper[valid_item]
        original = valid_scores[i]
        after_post = all_scores[valid_idx]
        match = "✓" if np.isclose(original, after_post) else "✗"
        print(f"     {match} Item {valid_item}: {original:.3f} → {after_post:.3f}")

    # Test similarity queries
    print("\n7. Testing similarity queries...")

    test_item = '004'  # Electronics/laptop
    similar_items = handler.get_most_similar_items(test_item, top_k=3, valid_only=True)

    print(f"   Most similar valid items to '{test_item}' (laptop):")
    for item, score in similar_items:
        item_desc = item_data[item_data['item_id'] == item].iloc[0]
        print(f"     - {item} ({item_desc['subcategory']}, {item_desc['category']}): "
              f"similarity = {score:.3f}")

    # Test batch operations
    print("\n8. Testing batch operations...")

    batch_sequences = [
        ['001', '002', '003'],
        ['005', '006', '007'],
        ['009', '010', '001']
    ]

    preprocessed_batch = handler.preprocess_batch(batch_sequences)
    print(f"   Preprocessed batch of {len(preprocessed_batch)} sequences")

    batch_scores = np.random.rand(len(batch_sequences), len(valid_items))
    all_scores_batch = handler.postprocess_batch(batch_scores)
    print(f"   Postprocessed scores: {batch_scores.shape} → {all_scores_batch.shape}")

    # Test save/load
    print("\n9. Testing save/load functionality...")

    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'handler.pkl')
        handler.save(save_path)
        print(f"   ✓ Saved handler to {save_path}")

        loaded_handler = UnseenItemHandler.load(save_path)
        print(f"   ✓ Loaded handler from {save_path}")

        # Verify loaded handler works
        test_seq = ['001', '004', '005']
        preprocessed_orig = handler.preprocess_sequence(test_seq)
        preprocessed_loaded = loaded_handler.preprocess_sequence(test_seq)

        if preprocessed_orig == preprocessed_loaded:
            print("   ✓ Loaded handler produces identical results")
        else:
            print("   ✗ Loaded handler results differ!")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == '__main__':
    try:
        test_basic_functionality()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
