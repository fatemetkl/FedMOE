import argparse
import logging
import os
import re

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_loss_from_log(run_folder_path: str, delete_error_files: bool = False) -> float | None:
    server_log_path = os.path.join(run_folder_path, "log.out")
    server_done_path = os.path.join(run_folder_path, "done.out")
    if not os.path.exists(server_done_path):
        logger.info(f"{run_folder_path} experiment is not completed, run again.")
        return None
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
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiment")

    parser.add_argument(
        "--directory_of_data",
        action="store",
        type=str,
        required=True,
        help="Path folder containing experimental data",
    )

    parser.add_argument(
        "--output_dir",
        action="store",
        type=str,
        required=True,
        help="Path to directory for the visualizations",
    )

    args = parser.parse_args()

    directory_of_data = args.directory_of_data
    output_dir = args.output_dir

    two_client_folder = os.path.join(directory_of_data, "experiment_2c")
    five_client_folder = os.path.join(directory_of_data, "experiment_5c")

    two_client_experiments = [
        os.path.join(two_client_folder, contents)
        for contents in os.listdir(two_client_folder)
        if "DS_Store" not in contents
    ]
    five_client_experiments = [
        os.path.join(five_client_folder, contents)
        for contents in os.listdir(five_client_folder)
        if "DS_Store" not in contents
    ]

    parse_parameter_re = r"gameT(\d+)_"

    two_client_min_mses = []
    for experiment in two_client_experiments:
        m = re.search(parse_parameter_re, str(experiment))
        min_mse = None
        for index in range(3):
            run_folder = os.path.join(experiment, f"Run{index+1}")
            mse = get_loss_from_log(run_folder)
            if mse is None:
                continue
            if mse > 5.0:
                continue
            if min_mse is None or min_mse > mse:
                min_mse = mse

        if min_mse is not None:
            two_client_min_mses.append((int(m.group(1)), min_mse))

    five_client_min_mses = []
    for experiment in five_client_experiments:
        m = re.search(parse_parameter_re, str(experiment))
        min_mse = None
        for index in range(3):
            run_folder = os.path.join(experiment, f"Run{index+1}")
            mse = get_loss_from_log(run_folder)
            if mse is None:
                continue
            if mse > 5.0:
                continue
            if min_mse is None or min_mse > mse:
                min_mse = mse

        if min_mse is not None:
            five_client_min_mses.append((int(m.group(1)), min_mse))

    two_client_min_mses = sorted(two_client_min_mses, key=lambda x: x[0])
    five_client_min_mses = sorted(five_client_min_mses, key=lambda x: x[0])
    two_client_parameter_values = [param for (param, _) in two_client_min_mses]
    five_client_parameter_values = [param for (param, _) in five_client_min_mses]
    two_client_mses = [mse for (_, mse) in two_client_min_mses]
    five_client_mses = [mse for (_, mse) in five_client_min_mses]

    plt.rcParams["figure.figsize"] = [10, 6]
    ax = plt.figure().gca()
    ax.set_yscale("log")

    plt.scatter(two_client_parameter_values, two_client_mses, label="2 Clients", s=80)
    plt.scatter(five_client_parameter_values, five_client_mses, label="5 Clients", s=80)

    title_font = {"family": "helvetica", "weight": "bold", "size": 30}
    axis_font = {"family": "helvetica", "weight": "bold", "size": 24}
    plt.xticks(fontname="helvetica", fontsize=20, fontweight="bold")
    plt.yticks(fontname="helvetica", fontsize=20, fontweight="bold")
    plt.xticks(fontname="helvetica", fontsize=20, minor=True, fontweight="bold")
    plt.yticks(fontname="helvetica", fontsize=20, minor=True, fontweight="bold")
    plt.xlabel("Nash Game Lookback Length", fontdict=axis_font)
    plt.ylabel("Minimum MSE", fontdict=axis_font)
    plt.title("Minimum MSE for BoC Exchange Rate Series", fontdict=title_font)

    plt.legend(prop={"family": "helvetica", "weight": "bold", "size": 24}, loc="lower right", labelspacing=0)
    plt.tight_layout(pad=1)

    plt.savefig(os.path.join(output_dir, "min_mses_over_lookback.png"))

    plt.close()
