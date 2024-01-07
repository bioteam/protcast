import argparse
import yaml
import sys
from pathlib import Path
from simple_slurm import Slurm

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))


def main():
    """repeat_slurm_job.py

    >>> slurm.squeue.jobs
    {6030566: {'JOBID': '6030566', 'NAME': 'get_simple_dataset_files', 'ST': 'R',
    'TIME': '16:41', 'TIME_LEFT': '1-23:43:19', 'NODES': '1', 'CPUS': '56',
    'MIN_MEMORY': '0', 'TRES_PER_NODE': 'N/A', 'NODELIST(REASON)': 'c208-020'}}
    ...
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--name",
        help="Job name",
        default="get_simple_dataset_files",
    )
    args = parser.parse_args()

    slurm = Slurm(**yaml.safe_load(open(".slurm_default.yml", "r")))
    slurm.squeue.update_squeue()

    for jobid in slurm.squeue.jobs.keys():
        if slurm.squeue.jobs[jobid]["NAME"] is args.name:
            if slurm.squeue.jobs[jobid]["ST"] == "PD":
                slurm.scancel(jobid)


if __name__ == "__main__":
    main()
