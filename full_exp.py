import os
import torch
import gc
import argparse

config_file_path = "/workspace/data-mining/DataMining_RecBLR/config.yaml"

def clean_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")
        
def gen_config(
    bd_lru_only: bool = False,
    disable_conv1d: bool = False,
    disable_ffn: bool = False,
    num_layers: int = 2,
    num_epochs: int = 100,
    file_path: str = config_file_path,
    dataset: str = "amazon-beauty"
) -> None:
    yaml_content = f"""
gpu_id: '0'

# RecBLR architecture flags
bd_lru_only: {bd_lru_only}
disable_conv1d: {disable_conv1d}
disable_ffn: {disable_ffn}

# RecBLR settings
hidden_size: 64
num_layers: {num_layers}
dropout_prob: 0.2
loss_type: 'CE'
expand: 2
d_conv: 4

# dataset settings
dataset: {dataset}
MAX_ITEM_LIST_LENGTH: 200    

USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
load_col:
    inter: [user_id, item_id, timestamp]

user_inter_num_interval: "[5,inf)"
item_inter_num_interval: "[5,inf)"

# training settings
epochs: {num_epochs}
train_batch_size: 2048
learner: adam
learning_rate: 0.001
eval_step: 1
stopping_step: 10
train_neg_sample_args: ~

# evalution settings
metrics: ['Hit', 'NDCG', 'MRR']
valid_metric: NDCG@10
eval_batch_size: 4096
weight_decay: 0.0
topk: [10, 20]
"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with open(file_path, 'w') as f:
            f.write(yaml_content)
        print(f"Successfully wrote configuration to {file_path}")
    except Exception as e:
        print(f"Error writing file {file_path}: {e}")

def run(cmd):
    print(f"Running: {cmd}")
    os.system(cmd)

def run_comp_default(model, epochs, dataset):
    gen_config(num_epochs=epochs, dataset=dataset)
    run(f"python run.py --model {model}")

def run_comp_1layer(model, epochs, dataset):
    gen_config(num_layers=1, num_epochs=epochs, dataset=dataset)
    run(f"python run.py --model {model}")

def run_comp_bdlru(model, epochs, dataset):
    gen_config(bd_lru_only=True, num_epochs=epochs, dataset=dataset)
    run(f"python run.py --model {model}")

def run_comp_noconv(model, epochs, dataset):
    gen_config(disable_conv1d=True, num_epochs=epochs, dataset=dataset)
    run(f"python run.py --model {model}")

def run_comp_noff(model, epochs, dataset):
    gen_config(disable_ffn=True, num_epochs=epochs, dataset=dataset)
    run(f"python run.py --model {model}")

def run_comp_all(model, epochs, dataset):
    run_comp_default(model, epochs, dataset)
    run_comp_1layer(model, epochs, dataset)
    run_comp_bdlru(model, epochs, dataset)
    run_comp_noconv(model, epochs, dataset)
    run_comp_noff(model, epochs, dataset)

def run_experiment_model(epochs, dataset):
    gen_config(num_epochs=epochs, dataset=dataset)
    run("python run.py --model R")
    gen_config(num_epochs=epochs, dataset=dataset)
    run("python run.py --model B")
    gen_config(num_epochs=epochs, dataset=dataset)
    run("python run.py --model S")

def run_experiment_unseen(epochs, dataset, mode):
    gen_config(num_epochs=epochs, dataset=dataset)
    run(f"python run_with_unseen.py --mode {mode}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=False, choices=["r", "b", "s"])
    parser.add_argument("--exp", required=True, choices=["comp", "model", "unseen"])
    parser.add_argument("--mode", choices=["default", "1layer", "bdlru", "noconv", "noff", "all", "none", "pre"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="amazon-beauty")
    args = parser.parse_args()

    if args.exp == "comp":
        model = args.model.upper()
        if not args.mode:
            print("Specify mode for --exp comp: default, 1layer, bdlru, noconv, noff, all")
            return
        if args.mode == "default":
            run_comp_default(model, args.epochs, args.dataset)
        elif args.mode == "1layer":
            run_comp_1layer(model, args.epochs, args.dataset)
        elif args.mode == "bdlru":
            run_comp_bdlru(model, args.epochs, args.dataset)
        elif args.mode == "noconv":
            run_comp_noconv(model, args.epochs, args.dataset)
        elif args.mode == "noff":
            run_comp_noff(model, args.epochs, args.dataset)
        elif args.mode == "all":
            run_comp_all(model, args.epochs, args.dataset)

    elif args.exp == "model":
        model = args.model.upper()
        run_experiment_model(args.epochs, args.dataset)

    elif args.exp == "unseen":
        if args.mode is None:
            print("Running all modes")
            for mode in ["none", "pre"]:
                print(f"\n{'='*80}\nRUNNING MODE: {mode}\n{'='*80}\n")
                run_experiment_unseen(args.epochs, args.dataset, mode)
                clean_cache()
        elif args.mode in ["none", "pre"]:
            run_experiment_unseen(args.epochs, args.dataset, args.mode)
        else:
            print("For --exp unseen, valid modes are: none, pre")
            return

if __name__ == "__main__":
    main()