import subprocess
from .experiment import *
from hashlib import sha256


def get_title(req) -> str:
    return f'{req.function_name}-batch{req.batch_size}-k{req.options.k}-parallel{req.options.default_parallelism}-TTA{req.options.goal_accuracy}'


def get_hash(title: str) -> str:
    """Given the experiment title return a hash so experimets with the same params
    can be identified between replications"""
    return sha256(title.encode('utf-8')).hexdigest()[:16]


def check_stderr(res: subprocess.CompletedProcess):
    """Checks whether the executed command returned an error and exists if that is the case"""
    if len(res.stderr) == 0:
        return
    print("error running command", res.args, res.stderr.decode())
    raise Exception


def create_function(name: str, file: str):
    """Creates a function in kubeml"""

    command = f"kubeml fn create --name {name} --code {file}"
    print("Creating function", name)

    res = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    check_stderr(res)

    print("function created")
