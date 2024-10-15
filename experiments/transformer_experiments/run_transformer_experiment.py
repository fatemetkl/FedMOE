import argparse
import logging
import os
import torch

import random
from typing import Any, Dict

from experiments.utils import load_config, load_data

from fedmoe.client_manager import PreTrainingClientManager
from fedmoe.game import TransformerGame
from fedmoe.metrics import RMSEMetric
from fedmoe.server import Server


def main(
    config: Dict[str, Any],
    results_dir: str,
    hidden_dim: int,
    T: int,
    alpha: float,
    gamma: float,
    sigma: float,
    K: float,
    eta: float,
    data_loader_num_samples: int,
    data_loader_batch_size: int,
    pre_training_epochs: int,
    pre_training_learning_rate: float,
) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Configuration loaded")

    assert os.path.exists(results_dir), "Error: result path does not exists"

    # Load data
    data_object = load_data(config["data"], config["total_rounds"] + 1)
    train_data_loader = data_object.get_dataloader(
        num_samples=data_loader_num_samples, batch_size=data_loader_batch_size, shuffle=True
    )

    client_manager = PreTrainingClientManager(
        config["num_clients"],
        data_object.input_matrix,
        T,
        hidden_dim,
        alpha,
        gamma,
        pre_training_dataloader=train_data_loader,
        pre_training_epochs=pre_training_epochs,
        pre_training_learning_rate=pre_training_learning_rate,
        target_sequence=data_object.target_matrix,
    )

    game = TransformerGame(
        client_manager.clients,
        sync_freq=T,
        z_dim=hidden_dim,
    )
    logger.info("Transformer clients initiated")

    # Run the server
    server = Server(
        sync_freq=T,
        client_manager=client_manager,
        game=game,
        metrics=[RMSEMetric("RMSE")],
        kappa=K,
        eta=eta,
    )
    logger.info("Server initiated")

    final_metric_value = server.fit(config["total_rounds"], config["have_sync"], config["update_last_Y_sync"])
    print("Final metric value:", "\n", final_metric_value["server - server_predictions - RMSE"])
    # Plot or save server predictions and the input data sequence
    plot_info = {
        "num_clients": config["num_clients"],
        "T": T,
        "d_z": hidden_dim,
        "alpha": alpha,
        "gamma": gamma,
        "sigma": sigma,
        "pre_training_num_samples": data_loader_num_samples,
        "pre_training_batch_size": data_loader_batch_size,
        "pre_training_epochs": pre_training_epochs,
        "pre_training_learning_rate": pre_training_learning_rate,
    }

    if config["save_plot"]:
        data_object.visualize(server.server_outputs, f"{results_dir}/plot.png", plot_info=plot_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiment")
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file.",
        default="experiments/transformer_experiments/config.yaml",
    )
    parser.add_argument(
        "--result_dir",
        action="store",
        type=str,
        help="Path to results directory.",
        default="results/experiment",
    )
    parser.add_argument(
        "--hidden_dim",
        action="store",
        type=int,
        help="d_z value.",
        default=5,
    )
    parser.add_argument(
        "--alpha",
        action="store",
        type=float,
        help="Alpha value.",
        default=0.01,
    )
    parser.add_argument(
        "--gamma",
        action="store",
        type=float,
        help="Gamma value.",
        default=0.01,
    )
    parser.add_argument(
        "--sigma",
        action="store",
        type=float,
        help="Sigma value.",
        default=0.01,
    )
    parser.add_argument(
        "--K",
        action="store",
        type=float,
        help="Kappa value for server mixture.",
        default=1.0,
    )
    parser.add_argument(
        "--eta",
        action="store",
        type=float,
        help="Eta value for server mixture.",
        default=1.0,
    )
    parser.add_argument(
        "--T",
        action="store",
        type=int,
        help="Sync step value.",
        default=5,
    )
    parser.add_argument(
        "--data_loader_num_samples",
        action="store",
        type=int,
        help="The size of the dataset used for transformer pre-training.",
        default=100,
    )
    parser.add_argument(
        "--data_loader_batch_size",
        action="store",
        type=int,
        help="Batch size used to create the data loader for transformer pre-training.",
        default=5,
    )
    parser.add_argument(
        "--pre_training_epochs",
        action="store",
        type=int,
        help="The number of epochs used for transformer pre-training.",
        default=2,
    )
    parser.add_argument(
        "--pre_training_learning_rate",
        action="store",
        type=float,
        help="The learning rate used in transformer pre-training.",
        default=0.01,
    )
    parser.add_argument(
        "--random_seed",
        action="store",
        type=int,
        help="Random seed value.",
        default=2024,
    )

    args = parser.parse_args()
    config = load_config(args.config_path)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    main(
        config,
        args.result_dir,
        args.hidden_dim,
        args.T,
        args.alpha,
        args.gamma,
        args.sigma,
        args.K,
        args.eta,
        args.data_loader_num_samples,
        args.data_loader_batch_size,
        args.pre_training_epochs,
        args.pre_training_learning_rate,
    )
