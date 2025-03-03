import argparse
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
            loss = float(line_to_convert)
            return loss
        except Exception:
            logger.info(f"{run_folder_path} file did not run completely due to error, check log file.")
            # Returning max loss
            if delete_error_files:
                logger.info(f"Deleting the run log with error {server_done_path}.")
                os.remove(server_done_path)
            return 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiment")

    parser.add_argument(
        "--directory_of_runs",
        action="store",
        type=str,
        required=True,
        help="Path to experiment containing folders Run1, Run2, Run3",
    )

    args = parser.parse_args()

    directory_of_runs = args.directory_of_runs
    total = 0.0
    for index in range(3):
        run_folder = os.path.join(directory_of_runs, f"Run{index+1}")
        total += get_loss_from_log(run_folder)

    logger.info(f"{total/3.0}")
