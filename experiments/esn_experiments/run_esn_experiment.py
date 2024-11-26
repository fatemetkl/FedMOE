import argparse
import logging
import os
import random
from typing import Any, Dict, List

import torch

from experiments.utils import load_config, load_data, save_to_json
from fedmoe.client_manager import ClientManager
from fedmoe.clients.client import ClientType
from fedmoe.game import EchoStateGame
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
) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Configuration loaded")

    assert os.path.exists(results_dir), "Error: result path does not exists"

    # Load data
    data_object = load_data(config["data"], config["total_rounds"] + 1)

    client_manager = ClientManager(
        ClientType.ESN,
        config["num_clients"],
        data_object.input_matrix,
        T,
        hidden_dim,
        alpha,
        gamma,
        sigma,
        data_object.target_matrix,
    )

    game = EchoStateGame(
        client_manager.clients,
        sync_freq=T,
        z_dim=hidden_dim,
    )
    logger.info("ESN clients initiated")

    # Run the server
    server = Server(
        total_game_steps=T,
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
    }

    tensors_to_save: Dict[str, List[torch.Tensor]] = {}

    if config["save_server_prediction"]:
        data_object.visualize_server_prediction(
            server.server_outputs, f"{results_dir}/server_pred_plot.png", plot_info=plot_info
        )
        tensors_to_save["server_prediction"] = server.server_outputs

    if config["save_input"]:
        data_object.visualize_input(f"{results_dir}/input_plot.png", plot_info=plot_info)
        # Converting a matrix to a list of tensors to avoid mypy errors.
        tensors_to_save["input"] = [row for row in data_object.input_matrix]

    if config["save_clients_predictions"]:
        data_object.visualize_clients_predictions(
            server.clients_predictions, plot_path=f"{results_dir}/client_predictions.png", plot_info=plot_info
        )
        tensors_to_save["clients_predictions"] = server.clients_predictions

    if config["save_mixture_weights"]:
        data_object.visualize_mixture_weights(
            server.mixture_weights, plot_path=f"{results_dir}/mixture_weights.png", plot_info=plot_info
        )
        tensors_to_save["mixture_weights"] = server.mixture_weights

    if config["dump_json"]:
        # Dump results and data in JSON
        save_to_json(tensors_to_save, path=f"{results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiment")
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file.",
        default="experiments/esn_experiments/config.yaml",
    )
    parser.add_argument(
        "--result_dir",
        action="store",
        type=str,
        help="Path to results directory.",
        default="results/esn_experiment",
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
        "--random_seed",
        action="store",
        type=int,
        help="Random seed value.",
        default=5,
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
    )
