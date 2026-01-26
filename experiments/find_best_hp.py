import argparse
import logging
import os

import numpy as np
import torch


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

torch.set_default_dtype(torch.float64)


def get_hp_folders(hp_sweep_dir: str) -> list[str]:
    paths_in_hp_sweep_dir = [os.path.join(hp_sweep_dir, contents) for contents in os.listdir(hp_sweep_dir)]
    return [hp_folder for hp_folder in paths_in_hp_sweep_dir if os.path.isdir(hp_folder)]


def get_run_folders(hp_dir: str) -> list[str]:
    run_folder_names = [folder_name for folder_name in os.listdir(hp_dir) if "Run" in folder_name]
    return [os.path.join(hp_dir, run_folder_name) for run_folder_name in run_folder_names]


def get_loss_from_log(run_folder_path: str, delete_error_files: bool = False) -> float:
    server_log_path = os.path.join(run_folder_path, "log.out")
    server_done_path = os.path.join(run_folder_path, "done.out")
    if not os.path.exists(server_done_path):
        logger.info(f"{run_folder_path} experiment is not completed, run again.")
        return 200
    with open(server_log_path, "r") as handle:
        files_lines = handle.readlines()
        line_to_convert = files_lines[-1].strip()
        try:
            return float(line_to_convert)
        except Exception:
            logger.info(f"{run_folder_path} file did not run completely due to error, check log file.")
            # Returning max loss
            if delete_error_files:
                logger.info(f"Deleting the run log with error {server_done_path}.")
                os.remove(server_done_path)
            return 100


def main(hp_sweep_dir: str, delete_error_files: bool = False) -> None:
    hp_folders = get_hp_folders(hp_sweep_dir)
    best_avg_loss: float | None = None
    best_folder = ""
    error_runs_count = 0
    not_completed_runs_count = 0
    completed_runs_count = 0
    for hp_folder in hp_folders:
        run_folders = get_run_folders(hp_folder)
        hp_losses = []
        for run_folder in run_folders:
            run_loss = get_loss_from_log(run_folder, delete_error_files)
            if run_loss == 100:
                error_runs_count += 1
            elif run_loss == 200:
                not_completed_runs_count += 1
            else:
                completed_runs_count += 1
            hp_losses.append(run_loss)
        current_avg_loss = float(np.mean(hp_losses))
        if best_avg_loss is None or current_avg_loss <= best_avg_loss:
            logger.info(f"Current Loss: {current_avg_loss} is lower than Best Loss: {best_avg_loss}")
            logger.info(f"Best Folder: {hp_folder}, Previous Best: {best_folder}")
            best_avg_loss = current_avg_loss
            best_folder = hp_folder
    logger.info(f"Completed runs count: {completed_runs_count}.")
    logger.info(f"Error runs count: {error_runs_count}.")
    logger.info(f"Not completed runs count (probably preempted)): {not_completed_runs_count}.")
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
    parser.add_argument(
        "--delete_error_files",
        action="store",
        type=bool,
        help="Pass True if you want to delete the done.out of error runs to run them again.",
        required=False,
    )

    args = parser.parse_args()

    logger.info(f"Hyperparameter Sweep Directory: {args.hp_sweep_dir}")
    main(args.hp_sweep_dir, args.delete_error_files)
