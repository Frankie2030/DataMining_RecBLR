import sys
import logging
from logging import getLogger
from io import StringIO
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from RecBLR import RecBLR
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
    config_files = ['configs/config_amazon_beauty.yaml',
                    'configs/config_amazon_sports.yaml',
                    'configs/config_amazon_apps.yaml',
                    'configs/config_yelp.yaml']
    for config_file in config_files:
        print(f"Running with config file: {config_file}")
        config = Config(model=RecBLR, config_file_list=[config_file])

        init_seed(config['seed'], config['reproducibility'])
        
        # Create a StringIO object to capture log output
        log_capture_string = StringIO()
        ch = logging.StreamHandler(log_capture_string)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        # logger initialization
        init_logger(config)
        logger = getLogger()
        logger.addHandler(ch) # Add the handler to the logger
        
        logger.info(sys.argv)
        logger.info(config)

        # dataset filtering
        dataset = create_dataset(config)
        logger.info(dataset)

        # dataset splitting
        train_data, valid_data, test_data = data_preparation(config, dataset)

        # model loading and initialization
        init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
        model = RecBLR(config, train_data.dataset).to(config['device'])
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
        log_contents = log_capture_string.getvalue()
        df = parse_log_text(log_contents)
        
        # Extract filename without extension for output prefix
        output_prefix = config_file.split('/')[-1].replace('.yaml', '_')
        generate_plots(df, output_prefix)
        df.to_csv(f"{output_prefix}training_metrics.csv", index=False)
        print(f"Metrics for {config_file} saved to {output_prefix}training_metrics.csv")
        print(f"Plots for {config_file} generated with prefix {output_prefix}")

        # Clean up logger for the next run
        logger.removeHandler(ch)
        ch.close()
        print("-" * 50) # Separator for readability between runs
