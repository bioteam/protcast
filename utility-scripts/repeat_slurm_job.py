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
        help="Job name",
        default="get_simple_dataset_files",
    )
    parser.add_argument(
        "-a",
        "--account",
        help="Account name",
        default="bosborne",
    )
    args = parser.parse_args()

    jobs = pyslurm.db.JobFilter(account=args.account)
