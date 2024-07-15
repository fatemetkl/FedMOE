import argparse
import logging
import os

# import random
from typing import Any, Dict

import torch

from experiments.utils import load_config, load_data, plot_sequence

# from fl4health.utils.metrics import Metric
from fedmoe.client_manager import ClientManager
from fedmoe.clients.client import ClientType
from fedmoe.game import RfnGame
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
    eta: int,
    experiment_setup: str,
) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Configuration loaded")

    assert os.path.exists(results_dir), "Error: result path does not exists"
    assert (
        config["client_type"] == ClientType.RFN.value
    ), "Error: this experiment file only runs Random Feature Example"

    # Load data
    data_sequence = load_data(config["data"], config["total_rounds"])

    client_manager = ClientManager(
        config["client_type"],
        config["num_clients"],
        data_sequence,
        T,
        hidden_dim,
        alpha,
        gamma,
        sigma,
    )

    game = RfnGame(
        client_manager.clients,
        sync_freq=T,
        z_dim=hidden_dim,
    )
    logger.info("RFN clients initiated")

    # Run the server
    server = Server(
        sync_freq=T,
        client_manager=client_manager,
        game=game,
        metrics=[RMSEMetric("RSME")],
        kappa=K,
        eta=eta,
    )
    logger.info("Server initiated")

    server.fit(config["total_rounds"], config["have_sync"], config["update_last_Y_sync"])

    # Plot or save server predictions and the input data sequence
    plot_info = {
        "num_clients": config["num_clients"],
        "T": T,
        "d_z": hidden_dim,
        "alpha": alpha,
        "gamma": gamma,
        "sigma": sigma,
    }

    if config["save_plot"]:
        server_preds = torch.Tensor([item.squeeze().detach() for item in server.server_outputs])
        plot_sequence(
            client_manager.common_target_sequence, server_preds, T, config["have_sync"], plot_info, results_dir
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiment")
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file.",
        default="experiments/rfn_experiments/config.yaml",
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
        type=int,
        help="Eta value for server mixture.",
        default=3,
    )
    parser.add_argument(
        "--T",
        action="store",
        type=int,
        help="Sync step value.",
        default=5,
    )
    parser.add_argument(
        "--experiment_setup",
        action="store",
        type=str,
        help="A string representing the value of hyper-parameters.",
        default="T5_alpha0.01_gamma0.01_sigma0.01_DZ5_K1_ETA3",
    )

    # seed = 2024
    # random.seed(seed)
    # torch.manual_seed(seed)
    args = parser.parse_args()
    config = load_config(args.config_path)
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
        args.experiment_setup,
    )
