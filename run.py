import sys
import logging
from logging import getLogger
import argparse
import os

from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from RecBLR import RecBLR
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run recommender models with specific configurations.')
    parser.add_argument('--model', type=str, default='R', choices=['B', 'R', 'S'],
                        help='Model to use: B for Bert4Rec, R for RecBLR (default: R), S for SASRec')
    args = parser.parse_args()

    if args.model == 'B':
        model_class = BERT4Rec
    elif args.model == 'R':
        model_class = RecBLR
    elif args.model == 'S':
        model_class = SASRec

    config_file = 'config.yaml'
    config = Config(model=model_class, config_file_list=[config_file])

    # Apply specific RecBLR architecture flags only when model is RecBLR
    if args.model != 'R':
        config['bd_lru_only'] = False
        config['disable_conv1d'] = False
        config['disable_ffn'] = False

    init_seed(config['seed'], config['reproducibility'])
    
    # Create a FileHandler to capture log output
    log_file_path = f"temp_run_log_{model_class.__name__}.log"
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.addHandler(file_handler) # Add the file handler to the logger
    
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = model_class(config, train_data.dataset).to(config['device'])
    logger.info(model)
    
    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=config["show_progress"]
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, show_progress=config["show_progress"]
    )
    
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
    generate_plots(df, output_prefix)
    df.to_csv(f"{output_prefix}training_metrics.csv", index=False)
    print(f"Metrics for {config_file} saved to {output_prefix}training_metrics.csv")
    print(f"Plots for {config_file} generated with prefix {output_prefix}")

    # Clean up logger and temporary log file
    logger.removeHandler(file_handler)
    file_handler.close()
    os.remove(log_file_path)
