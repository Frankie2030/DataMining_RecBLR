import pandas as pd
import os
from plot_utils import parse_log_text, generate_comparison_plots

if __name__ == '__main__':
    model_names = ["BERT4Rec", "SASRec", "RecBLR"]
    all_models_data = []

    for model_name in model_names:
        log_file_path = f"temp_run_log_{model_name}.log"
        
        if not os.path.exists(log_file_path):
            print(f"Error: Log file not found for {model_name} at {log_file_path}. Please ensure you have run `python run.py --model [B/S/R]` for each model to generate the log files.")
            continue

        with open(log_file_path, 'r') as f:
            log_contents = f.read()
        
        df = parse_log_text(log_contents)
        df['Model'] = model_name # Add model name to the DataFrame
        all_models_data.append(df)

    if all_models_data:
        combined_df = pd.concat(all_models_data, ignore_index=True)
        generate_comparison_plots(combined_df)
        print("All comparison plots generated successfully.")
    else:
        print("No data collected to generate comparison plots.")
