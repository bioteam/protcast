import argparse
import sys
import pyslurm
from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))


if __name__ == "__main__":
    """repeat_slurm_job.py"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--name",
        default="get_simple_dataset_files",
        help="Job name",
    )
    parser.add_argument(
        "-a",
        "--account",
        default="bosborne",
        help="Account name",
    )
    args = parser.parse_args()

    a = pyslurm.job()
    pending = a.find("job_state", "PENDING")
    # running = a.find("job_state", "RUNNING")
    # jobs = a.get()
