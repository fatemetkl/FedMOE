import argparse
import logging
import os
from typing import Any, Dict, List

import yaml
from torch.utils.data import DataLoader

from fedmoe.client_manager import ClientManager, PreTrainingClientManager
from fedmoe.clients.client import ClientType
from fedmoe.datasets.periodic_dataset import get_periodic_signal_sequence, load_periodic_dataloader
from fedmoe.game import EchoStateGame, Game, RfnGame, TransformerGame
from fedmoe.server import Server


def load_config(config_path: str) -> Dict[str, Any]:
    """Load Configuration Dictionary"""

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_pre_training_data(config: Dict[str, Any]) -> DataLoader:
    assert config["data"] == "periodic_signal"
    train_dataloader, val_dataloader, num_examples = load_periodic_dataloader(
        train_data_size=config["data_size"],
        val_data_size=0,
        batch_size=config["batch_size"],
        data_length=config["data_length"],
    )
    return train_dataloader


def main(config: Dict[str, Any], results_dir: str) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Configuration loaded")

    assert os.path.exists(results_dir), "Error: result path does not exists"

    # Load data
    if config["data"] == "periodic_signal":
        data_sequence = get_periodic_signal_sequence(config["n_samples"], config["data_length"])
    else:
        raise ValueError("dataset name is not valid.")
    logger.info("Dataset loaded")

    # Initiate client manager
    results: List[Dict[str, Any]] = []
    if config["client_type"] == ClientType.TRANSFORMER.value:
        # we need to do pre-training first, therefore we use pre-training client manager.
        data_loader = get_pre_training_data(config)
        client_manager: ClientManager = PreTrainingClientManager(
            config["client_type"],
            config["num_clients"],
            config["data_length"],
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
            config["data_length"],
            data_sequence,
            config["sync_freq"],
            config["d_z"],
            config["alpha"],
            config["gamma"],
        )
        if config["client_type"] == ClientType.RFN.value:
            game = RfnGame(
                client_manager.clients,
                sync_freq=config["sync_freq"],
                d_z=config["d_z"],
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
    server = Server(sync_freq=config["sync_freq"], client_manager=client_manager, game=game)
    final_metric_value = server.fit(config["total_rounds"])
    results.append(final_metric_value)
    with open(results_dir + "/" + config["experiment_name"] + ".txt", "w") as text_file:
        text_file.write(str(results))


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
    args = parser.parse_args()
    config = load_config(args.config_path)
    main(config, args.result_dir)
