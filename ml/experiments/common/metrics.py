import psutil
import GPUtil
from flask import Flask
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
import logging
from typing import List, Dict
import pickle
import pandas as pd
import os

from threading import Thread, Event

METRICS_FOLDER = os.path.abspath(os.path.dirname(__file__)) + '/../metrics'

FORMAT = '[%(asctime)s] %(levelname)-8s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger()

app = Flask(__name__)


@dataclass_json
@dataclass
class GpuStats:
    id: int
    name: str
    load: float
    mem_used: float
    mem_free: float


@dataclass_json
@dataclass
class MemoryStats:
    total: float
    free: float
    used: float
    percent: float


@dataclass_json
@dataclass
class CpuStats:
    percent: float


@dataclass_json
@dataclass
class SystemMetrics:
    exp_name: str
    gpu: Dict[str, List[GpuStats]] = field(default_factory=dict)
    mem: List[MemoryStats] = field(default_factory=list)
    cpu: List[CpuStats] = field(default_factory=list)

    def to_dataframe(self):
        """Converts the system metrics to a pandas df"""
        gpus = {}
        for id, metrics in self.gpu.items():
            gpus[f'gpu_{id}'] = [metrics]

        d = {
            'exp_name': self.exp_name,
            'cpu': [self.cpu],
            'mem': [self.mem],
            **gpus
        }

        return pd.DataFrame(d)


def get_cpu_usage() -> CpuStats:
    return CpuStats(percent=psutil.cpu_percent(1))


def get_memory_usage() -> MemoryStats:
    mem = psutil.virtual_memory()
    return MemoryStats(total=mem.total / 1e6, free=mem.free / 1e6, used=mem.used / 1e6, percent=mem.percent)


def get_gpu_usage() -> Dict[str, GpuStats]:
    stats = {}
    for gpu in GPUtil.getGPUs():
        id = gpu.id
        stats[id] = GpuStats(
            id=gpu.id,
            name=gpu.name,
            load=gpu.load,
            mem_used=gpu.memoryUsed,
            mem_free=gpu.memoryFree
        )
    return stats


def metrics_gathering_loop(name: str, pill: Event):
    """Loop gathering the metrics periodically"""
    metrics = SystemMetrics(exp_name=name)

    while not pill.wait(2):
        # logger.debug('getting metrics...')
        # get the metrics
        cpu = get_cpu_usage()
        gpu = get_gpu_usage()
        mem = get_memory_usage()

        # set the metrics in the object
        metrics.cpu.append(cpu)
        metrics.mem.append(mem)
        for k, v in gpu.items():
            if k in metrics.gpu:
                metrics.gpu[k].append(v)
            else:
                metrics.gpu[k] = [v]

    # when event sent, save the metrics to the the disk
    with open(f'{METRICS_FOLDER}/{metrics.exp_name}.pkl', 'wb') as f:
        pickle.dump(metrics.to_dataframe(), f)


running_job_pill: Event = None


# endpoints for the api
@app.route('/new/<string:id>', methods=['PUT'])
def new_task(id: str):
    """Starts a new training loop"""
    global running_job_pill
    logger.debug(f'computing new task with id {id}')

    # create the pill, set it as the current one and start the thread
    pill = Event()
    t = Thread(target=metrics_gathering_loop, args=(id, pill))
    running_job_pill = pill
    t.start()

    return "collection started", 200


@app.route('/finish', methods=['DELETE'])
def finish_task():
    """Finishes the currently running task"""
    global running_job_pill
    # try:
    running_job_pill.set()
    # except Exception as e:
    #     msg = f'error stopping thread, {e}'
    #     logger.error(msg)
    #     return msg, 500

    return 'saved experiment', 200


def start_api(output_folder=METRICS_FOLDER):
    # set the output folder if given
    global METRICS_FOLDER
    METRICS_FOLDER = output_folder
    print('Metrics using output folder', METRICS_FOLDER)

    logger.info('starting api...')
    app.run()


if __name__ == '__main__':
    app.run(debug=True)
