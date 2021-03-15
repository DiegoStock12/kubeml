import subprocess
from .experiment import *


def get_title(req: TrainRequest) -> str:
    return f'{req.model_type}-batch{req.batch_size}-k{req.options.K}-parallel{req.options.default_parallelism}-TTA{req.options.goal_accuracy}'


def check_stderr(res: subprocess.CompletedProcess):
    """Checks whether the executed command returned an error and exists if that is the case"""
    if len(res.stderr) == 0:
        return
    print("error running command", res.args, res.stderr.decode())
    exit(-1)


def create_function(name: str, file: str):
    """Creates a function in kubeml"""

    command = f"kubeml fn create --name {name} --code {file}"
    print("Creating function", name)

    res = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    check_stderr(res)

    print("function created")


