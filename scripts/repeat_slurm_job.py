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
        default="get_protcast_dataset_files",
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
    myjobs = a.find("account", args.account)

    # "wrap" specifies some command to execute before the script,
    # e.g. "module load def"
    # test_job = {
    #     "wrap": "sleep 3600",
    #     "job_name": "pyslurm_test_job",
    #     "cpus_per_task": 3,
    #     "partition": "abc",
    #     "script": "xyz",
    #     "nodes": 3,
    #     "time": "1:00:00",
    #     "output": "a.out",
    #     "error": "a.err"
    # }
    # test_job_id = pyslurm.job().submit_batch_job(test_job)
    # test_job_search = pyslurm.job().find(name="name", val=test_job["job_name"])
    # test_job_info = pyslurm.job().find_id(test_job_id)
    # rc = pyslurm.slurm_kill_job(test_job_id, Signal=9, BatchFlag=pyslurm.KILL_JOB_BATCH)
