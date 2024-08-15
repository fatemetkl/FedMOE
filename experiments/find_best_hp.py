import argparse
import logging
import os
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_hp_folders(hp_sweep_dir: str) -> List[str]:
    paths_in_hp_sweep_dir = [os.path.join(hp_sweep_dir, contents) for contents in os.listdir(hp_sweep_dir)]
    return [hp_folder for hp_folder in paths_in_hp_sweep_dir if os.path.isdir(hp_folder)]


def get_run_folders(hp_dir: str) -> List[str]:
    run_folder_names = [folder_name for folder_name in os.listdir(hp_dir) if "Run" in folder_name]
    return [os.path.join(hp_dir, run_folder_name) for run_folder_name in run_folder_names]


def get_loss_from_log(run_folder_path: str) -> float:
    server_log_path = os.path.join(run_folder_path, "log.out")
    with open(server_log_path, "r") as handle:
        files_lines = handle.readlines()
        line_to_convert = files_lines[-1].strip()
        try:
            loss = float(line_to_convert)
            return loss
        except Exception:
            logger.info(f"{run_folder_path} file did not run completely due to error, check log file.")
            # Returning max loss
            return 100


def main(hp_sweep_dir: str) -> None:
    hp_folders = get_hp_folders(hp_sweep_dir)
    best_avg_loss: Optional[float] = None
    best_folder = ""
    for hp_folder in hp_folders:
        run_folders = get_run_folders(hp_folder)
        hp_losses = []
        for run_folder in run_folders:
            run_loss = get_loss_from_log(run_folder)
            hp_losses.append(run_loss)
        current_avg_loss = float(np.mean(hp_losses))
        if best_avg_loss is None or current_avg_loss <= best_avg_loss:
            logger.info(f"Current Loss: {current_avg_loss} is lower than Best Loss: {best_avg_loss}")
            logger.info(f"Best Folder: {hp_folder}, Previous Best: {best_folder}")
            best_avg_loss = current_avg_loss
            best_folder = hp_folder
    logger.info(f"Best Loss: {best_avg_loss}")
    logger.info(f"Best Folder: {best_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Holdout Global")
    parser.add_argument(
        "--hp_sweep_dir",
        action="store",
        type=str,
        help="Path to the artifacts of the hyper-parameter sweep script",
        required=True,
    )

    args = parser.parse_args()

    logger.info(f"Hyperparameter Sweep Directory: {args.hp_sweep_dir}")
    main(args.hp_sweep_dir)
