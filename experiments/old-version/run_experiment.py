import argparse
import logging
import os
import random
from typing import Any, Dict

import matplotlib.pyplot as plt
import torch
import yaml

# from fl4health.utils.metrics import Metric
from torch.utils.data import DataLoader

from fedmoe.client_manager import ClientManager, PreTrainingClientManager
from fedmoe.clients.client import ClientType
from fedmoe.datasets.logistic_map_dataset import get_logistic_map_sequence, load_logistic_map_dataloader
from fedmoe.datasets.periodic_dataset import get_periodic_signal_sequence, load_periodic_dataloader
from fedmoe.game import EchoStateGame, Game, RfnGame, TransformerGame
from fedmoe.metrics import RMSEMetric
from fedmoe.server import Server


def plot_sequence(
    sequence1: torch.Tensor,
    sequence2: torch.Tensor,
    T: int,
    config: Dict[str, Any],
) -> None:
    plt.plot(sequence1, label="data", color="gray", alpha=0.5)
    plt.plot(sequence2, label="pred", color="blue", alpha=0.5, linewidth=2)

    if config["have_sync"]:
        T_indices = [i * T for i in range(1, int((sequence2.size(0) - 1) / T) + 1)]
        T_values = [sequence2[i] for i in T_indices]
        plt.scatter(T_indices, T_values, color="r", marker="o", label="T")

    fixed_variables = {
        "num_clients": config["num_clients"],
        "T": config["sync_freq"],
        "d_z": config["d_z"],
        "alpha": config["alpha"],
        "gamma": config["gamma"],
        "sigma": config["sigma"],
    }

    text_content = "\n".join([f"{key}: {value}" for key, value in fixed_variables.items()])
    plt.text(
        0.05,
        0.95,
        text_content,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Experiment data")
    plt.ylim((-2, 2))
    plt.legend()
    plt.show()


def plot_results(results: torch.Tensor, final_value: float, T: int) -> None:
    plt.plot(results, label="Results", color="gray", alpha=0.5)
    # Add a dot on each data point
    plt.scatter(range(results.size(0)), results, color="b", marker="o")
    # Highlight synchronization steps in red
    highlighted_indices = [i * T for i in range(int((results.size(0)) / T) + 1)]
    highlighted_values = torch.Tensor([results[i] for i in highlighted_indices])
    plt.scatter(highlighted_indices, highlighted_values, color="r", marker="o", label="Highlighted Points")

    plt.axhline(y=final_value, color="r", linestyle="--", label="Final value line")
    plt.xlabel("Time")
    plt.ylabel("Metric Value")
    plt.title("Experiment Results")
    plt.xticks(highlighted_indices)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.ylim((0, 2))
    plt.legend()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_pre_training_data(config: Dict[str, Any]) -> DataLoader:
    if config["data"] == "periodic_signal":
        train_dataloader, val_dataloader, num_examples = load_periodic_dataloader(
            train_data_size=config["data_size"],
            val_data_size=0,
            batch_size=config["batch_size"],
            data_length=config["total_rounds"] + 1,
        )
    elif config["data"] == "logistic_map":
        train_dataloader, val_dataloader, num_examples = load_logistic_map_dataloader(
            train_data_size=config["data_size"],
            val_data_size=0,
            batch_size=config["batch_size"],
            data_length=config["total_rounds"] + 1,
        )
    return train_dataloader


def main(config: Dict[str, Any], results_dir: str) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Configuration loaded")

    assert os.path.exists(results_dir), "Error: result path does not exists"

    # Load data
    if config["data"] == "periodic_signal":
        data_sequence = get_periodic_signal_sequence(config["n_samples"], config["total_rounds"] + 1)
        #  visualize data
        logger.info("Dataset loaded")
    elif config["data"] == "logistic_map":
        data_sequence = get_logistic_map_sequence(config["n_samples"], config["total_rounds"] + 1)
        logger.info("Dataset loaded")
    elif config["data"] == "linear_line":
        data_sequence = torch.Tensor([0.5 for i in range(config["total_rounds"] + 1)])
        # print(data_sequence)
    else:
        raise ValueError("dataset name is not valid.")

    # Initiate client manager
    if config["client_type"] == ClientType.TRANSFORMER.value:
        # we need to do pre-training first, therefore we use pre-training client manager.
        data_loader = get_pre_training_data(config)
        logger.info("Pre-training data loaded")
        client_manager: ClientManager = PreTrainingClientManager(
            config["client_type"],
            config["num_clients"],
            data_sequence,
            config["sync_freq"],
            config["d_z"],
            config["alpha"],
            config["gamma"],
            data_loader,
            config["pre_training_epochs"],
            config["pre_training_lr"],
        )
        game: Game = TransformerGame(
            client_manager.clients,
            sync_freq=config["sync_freq"],
            d_z=config["d_z"],
        )
        logger.info("Transformer clients initiated")
    else:
        client_manager = ClientManager(
            config["client_type"],
            config["num_clients"],
            data_sequence,
            config["sync_freq"],
            config["d_z"],
            config["alpha"],
            config["gamma"],
            config["sigma"],
        )
        if config["client_type"] == ClientType.RFN.value:
            game = RfnGame(
                client_manager.clients,
                sync_freq=config["sync_freq"],
                z_dim=config["d_z"],
            )
            logger.info("RFN clients initiated")
        elif config["client_type"] == ClientType.ESN.value:
            game = EchoStateGame(
                client_manager.clients,
                sync_freq=config["sync_freq"],
                d_z=config["d_z"],
            )
            logger.info("ESN clients initiated")
        else:
            print("Experiment terminated: not a valid client type")
    # Run the server
    server = Server(
        sync_freq=config["sync_freq"],
        client_manager=client_manager,
        game=game,
        metrics=[RMSEMetric("RSME")],
        kappa=config["K"],
        eta=config["eta"],
    )

    final_metric_value = server.fit(config["total_rounds"], config["have_sync"], config["update_last_Y_sync"])

    # plot server predictions and the input data sequence
    server_preds = torch.Tensor([item.squeeze().detach() for item in server.server_outputs])
    plot_sequence(client_manager.common_target_sequence, server_preds, config["sync_freq"], config)

    with open(results_dir + "/" + config["experiment_name"] + ".txt", "w") as file:
        file.write(str(final_metric_value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiment")
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file.",
        default="config.yaml",
    )
    parser.add_argument(
        "--result_dir",
        action="store",
        type=str,
        help="Path to results directory.",
        default="results/experiment",
    )
    seed = 2024
    random.seed(seed)
    torch.manual_seed(seed)
    args = parser.parse_args()
    config = load_config(args.config_path)
    main(config, args.result_dir)
