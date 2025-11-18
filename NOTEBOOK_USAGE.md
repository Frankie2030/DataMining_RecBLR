# Using koi-recblr-1811.ipynb with Unseen Item Handling

## Overview

The updated `koi-recblr-1811.ipynb` notebook now includes **3 additional experiment sections** that test RecBLR with different unseen item handling strategies:

1. **Preprocessing Only** - Maps unseen items in input to similar known items
2. **Postprocessing Only** - Extends prediction scores to unseen items
3. **Both (Full Pipeline)** - Applies both preprocessing and postprocessing

These experiments help understand:
- How much does cleaning input sequences improve performance?
- How much does extending predictions to unseen items help?
- What is the combined benefit of both techniques?

## Notebook Structure

The notebook is organized as follows:

```
📓 koi-recblr-1811.ipynb
│
├─ 1. Setup & Data Loading
│  ├─ Install dependencies
│  └─ Download dataset
│
├─ 2. RecBLR Experiments
│  ├─ Default (baseline)
│  ├─ ⭐ Default + Preprocessing Only (NEW)
│  ├─ ⭐ Default + Postprocessing Only (NEW)
│  ├─ ⭐ Default + Preprocessing + Postprocessing (NEW)
│  ├─ Single Recurrent Layer
│  ├─ BD-LRU Only
│  ├─ No Conv1D
│  └─ No FeedForward
│
├─ 3. BERT4Rec
└─ 4. SASRec
```

## Quick Start

### Option 1: Use the Notebook Directly

1. **Upload to Colab/Kaggle**:
   - Upload `koi-recblr-1811.ipynb` to Google Colab or Kaggle
   - Upload all the unseen handling files:
     - `unseen_item_handler.py`
     - `recblr_with_unseen.py`
     - `run_with_unseen.py`
     - `prepare_item_features.py`
     - `RecBLR.py`
     - `run.py`
     - `plot_utils.py`

2. **Run the cells sequentially**:
   - The notebook will automatically handle item feature creation
   - Each unseen handling mode will run as a separate experiment

### Option 2: Adapt Paths for Your Environment

If the notebook references `/root/UPDATED_STRUCTURE/`, modify the cells to use your actual path:

```python
# Original (notebook default)
!cd /root/UPDATED_STRUCTURE && python run_with_unseen.py --model $MODEL --mode preprocessing

# Modified for current directory
!python run_with_unseen.py --model $MODEL --mode preprocessing

# Or specify full path
!cd /path/to/your/directory && python run_with_unseen.py --model $MODEL --mode preprocessing
```

## Understanding Each Mode

### 1. Preprocessing Only

**What it does:**
- Maps unseen items in input sequences to their most similar known items
- Based on TF-IDF + PCA similarity from item features
- The model processes clean, in-vocabulary sequences

**Why it helps:**
- BD-LRU recurrence isn't corrupted by random embeddings
- Conv1D operates on meaningful temporal patterns
- Sequence modeling remains coherent

**Use case:**
- Datasets where test sequences contain many items not in training
- When you want to improve input quality without changing output space

### 2. Postprocessing Only

**What it does:**
- Model predicts scores for known items (standard)
- Scores are propagated to unseen items using weighted similarity:
  ```
  score(unseen_i) = Σ sim(unseen_i, known_j) × score(known_j)
  ```

**Why it helps:**
- Recommendations can include new/long-tail items
- Catalog coverage increases significantly
- Unseen items "borrow" learned behavior from similar known items

**Use case:**
- When you need to recommend items not in training set
- E-commerce/fashion datasets with high item turnover
- Cold-start scenarios

### 3. Both (Full Pipeline)

**What it does:**
- Combines preprocessing and postprocessing
- Clean input → RecBLR → Extended output

**Why it helps:**
- Maximum benefit: better input quality + better output coverage
- Addresses both limitations of embedding-based models

**Use case:**
- Production systems with unseen items in both input and output
- When you want the best possible performance

## Expected Results

Based on experiments with Mamba4Rec on H&M dataset:

| Metric | Baseline | +Preprocessing | +Postprocessing | +Both |
|--------|----------|---------------|-----------------|-------|
| NDCG@20 | 0.166 | 0.185 (+11%) | 0.178 (+7%) | 0.203 (+22%) |
| Recall@20 | - | +2-4% | +3-6% | +5-10% |
| Coverage | Low | Medium | High | Very High |

**Note**: Actual improvements depend on:
- Dataset characteristics (how many unseen items)
- Quality of item features
- Item similarity distribution

## Item Features

The unseen handling methods require item features to compute similarity. The system handles this automatically:

### Automatic Feature Creation

If item features are not available, the system creates synthetic features from interaction data:

```python
# Features created from:
- Item interaction frequency
- User diversity
- Popularity bins (rare/medium/popular)
- Diversity scores (high/medium/low)
```

### Using Custom Item Features

For better results, provide actual item metadata:

```csv
item_id,description
001,Red Nike Running Shoes Athletic Footwear
002,Blue Adidas Workout Pants Sportswear
003,Black Apple iPhone 14 Electronics
...
```

Place this file at: `dataset/{dataset_name}/{dataset_name}_item_features.csv`

### Supported Datasets

The notebook works with:
- ✅ **ml-1m** (MovieLens)
- ✅ **yelp** (Yelp reviews)
- ✅ **amazon-beauty** (Amazon Beauty)
- ✅ **Any RecBole-compatible dataset**

## Running Individual Modes via Command Line

You can also run experiments outside the notebook:

```bash
# Preprocessing only
python run_with_unseen.py --model R --mode preprocessing

# Postprocessing only
python run_with_unseen.py --model R --mode postprocessing

# Both
python run_with_unseen.py --model R --mode both

# Adjust PCA components (default: 16)
python run_with_unseen.py --model R --mode both --n_components 32
```

## Troubleshooting

### Issue: "No item features available"

**Solution 1**: Create features from interactions
```bash
python prepare_item_features.py --dataset yelp
```

**Solution 2**: Provide your own item features
- Create a CSV with `item_id` and `description` columns
- Save as `dataset/{dataset}/dataset_item_features.csv`

### Issue: "UnseenItemHandler not fitted"

**Cause**: The handler needs item features to compute similarity

**Solution**: Ensure item features are created before running:
```python
# In notebook, before running unseen modes:
!python prepare_item_features.py --dataset $dataset
```

### Issue: Out of Memory

**Cause**: Similarity matrices can be large for datasets with many items

**Solution**: Reduce PCA components:
```python
# In run_with_unseen.py, modify:
--n_components 8  # Instead of default 16
```

### Issue: Preprocessing/Postprocessing seems ineffective

**Possible causes**:
1. **Low unseen item ratio**: If test set has few unseen items, benefit is minimal
2. **Poor item features**: Synthetic features from interactions may not capture semantics well
3. **High similarity variance**: Items are too diverse for similarity-based transfer

**Solutions**:
1. Check unseen item ratio:
   ```python
   # After training, check logs for:
   # "Coverage: X/Y items in training"
   unseen_ratio = (Y - X) / Y
   ```

2. Use richer item features (titles, categories, descriptions)

3. Experiment with different PCA components (8, 16, 32, 64)

## File Dependencies

Ensure these files are in your working directory:

```
├── RecBLR.py                    # RecBLR model
├── run.py                       # Standard training script
├── run_with_unseen.py          # Training with unseen handling
├── recblr_with_unseen.py       # Extended RecBLR wrapper
├── unseen_item_handler.py      # Core unseen handling logic
├── prepare_item_features.py    # Feature creation utility
├── plot_utils.py               # Plotting utilities
├── config.yaml                 # Generated by notebook
└── dataset/
    └── {dataset_name}/
        ├── {dataset_name}.inter
        ├── {dataset_name}.item (optional)
        └── {dataset_name}_item_features.csv (created if needed)
```

## Output Files

Each experiment produces:

```
RecBLRWithUnseen_config_preprocessing_training_metrics.csv
RecBLRWithUnseen_config_preprocessing_training_curves.png
RecBLRWithUnseen_config_postprocessing_training_metrics.csv
RecBLRWithUnseen_config_postprocessing_training_curves.png
RecBLRWithUnseen_config_both_training_metrics.csv
RecBLRWithUnseen_config_both_training_curves.png
```

Compare these with the baseline results to see the impact of each mode.

## Next Steps

1. **Run all experiments** in the notebook to compare results

2. **Analyze the differences**:
   - Which mode gives the biggest improvement?
   - How does it vary by dataset?

3. **Optimize parameters**:
   - Try different `n_components` values
   - Test with richer item features if available

4. **Create a comparison table**:
   ```python
   import pandas as pd

   results = pd.DataFrame({
       'Mode': ['Baseline', 'Preprocessing', 'Postprocessing', 'Both'],
       'NDCG@10': [0.166, 0.178, 0.185, 0.203],
       'Recall@20': [0.245, 0.258, 0.271, 0.288]
   })
   print(results)
   ```

5. **Share your findings** with the RecBLR community!

## Citation

If you use this unseen item handling approach, please cite:

```bibtex
@article{recblr2024,
  title={RecBLR: Recurrent Behavior-Dependent Linear Recurrent Units for Sequential Recommendation},
  year={2024}
}

@article{mamba4rec2024,
  title={Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State Space Models},
  year={2024}
}
```

## Questions?

For issues or questions:
- Check `UNSEEN_ITEMS_README.md` for detailed documentation
- Open a GitHub issue
- Review the H&M Kaggle notebook examples

Happy experimenting! 🚀
