import pandas as pd
import re
import matplotlib.pyplot as plt
import logging

# Define regex patterns
re_train = re.compile(r"epoch (\d+) training .*? train loss: ([\d\.]+)")
re_eval = re.compile(r"epoch (\d+) evaluating .*? valid_score: ([\d\.]+)")
re_metrics = re.compile(r"hit@10 : ([\d\.]+) \s* hit@20 : ([\d\.]+) \s* ndcg@10 : ([\d\.]+) \s* ndcg@20 : ([\d\.]+) \s* mrr@10 : ([\d\.]+) \s* mrr@20 : ([\d\.]+)")

def parse_log_text(log_text):
    """
    Parses the log text to extract training and evaluation metrics.
    """
    log_lines = log_text.split('\n')
    data_dict = {}
    current_epoch = -1

    for line in log_lines:
        train_match = re_train.search(line)
        if train_match:
            epoch = int(train_match.group(1))
            current_epoch = epoch
            if epoch not in data_dict:
                data_dict[epoch] = {'Epoch': epoch}
            data_dict[epoch]['Train Loss'] = float(train_match.group(2))

        eval_match = re_eval.search(line)
        if eval_match:
            epoch = int(eval_match.group(1))
            current_epoch = epoch
            if epoch not in data_dict:
                data_dict[epoch] = {'Epoch': epoch}
            data_dict[epoch]['Valid Score'] = float(eval_match.group(2))

        metrics_match = re_metrics.search(line)
        if metrics_match:
            if current_epoch in data_dict:
                data_dict[current_epoch].update({
                    'Hit@10': float(metrics_match.group(1)),
                    'Hit@20': float(metrics_match.group(2)),
                    'NDCG@10': float(metrics_match.group(3)),
                    'NDCG@20': float(metrics_match.group(4)),
                    'MRR@10': float(metrics_match.group(5)),
                    'MRR@20': float(metrics_match.group(6))
                })

    final_data = [v for k, v in data_dict.items() if 'MRR@20' in v]
    df = pd.DataFrame(final_data)
    df = df.sort_values(by='Epoch')

    required_cols = ['Epoch', 'Train Loss', 'Valid Score', 'Hit@10', 'Hit@20', 'NDCG@10', 'NDCG@20', 'MRR@10', 'MRR@20']
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA
            
    return df

def generate_plots(df, output_prefix=""):
    """
    Generates and saves plots for training metrics.
    """
    if df.empty:
        logging.warning(f"No data to plot for prefix: {output_prefix}")
        return

    plt.figure(figsize=(10, 6))

    # Plot 1: Training Loss
    plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss', marker='o')
    plt.title(f'{output_prefix} Training Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}train_loss_plot.png')
    plt.clf()

    # Plot 2: Validation Score (NDCG@10)
    plt.plot(df['Epoch'], df['Valid Score'], label='Valid Score (NDCG@10)', marker='o', color='green')
    plt.title(f'{output_prefix} Validation Score (NDCG@10) vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Valid Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}valid_score_plot.png')
    plt.clf()

    # Plot 3: Hit Rate
    plt.plot(df['Epoch'], df['Hit@10'], label='Hit@10', marker='o')
    plt.plot(df['Epoch'], df['Hit@20'], label='Hit@20', marker='s')
    plt.title(f'{output_prefix} Hit Rate vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Hit Rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}hit_rate_plot.png')
    plt.clf()

    # Plot 4: NDCG
    plt.plot(df['Epoch'], df['NDCG@10'], label='NDCG@10', marker='o')
    plt.plot(df['Epoch'], df['NDCG@20'], label='NDCG@20', marker='s')
    plt.title(f'{output_prefix} NDCG vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('NDCG')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}ndcg_plot.png')
    plt.clf()

    # Plot 5: MRR
    plt.plot(df['Epoch'], df['MRR@10'], label='MRR@10', marker='o')
    plt.plot(df['Epoch'], df['MRR@20'], label='MRR@20', marker='s')
    plt.title(f'{output_prefix} MRR vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('MRR')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}mrr_plot.png')
    plt.clf()

    logging.info(f"Plots generated for {output_prefix}: train_loss_plot.png, valid_score_plot.png, hit_rate_plot.png, ndcg_plot.png, mrr_plot.png")
