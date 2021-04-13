"""trains the tensorflow experiments"""

import argparse
from common.metrics import start_api
from common.experiment import *

from multiprocessing import Process
import time

EPOCHS = 30
save_folder = './tests/tf'


def lenet(b: int):
    config = TfConfig(
        batch=b,
        epochs=EPOCHS,
        network='lenet'
    )

    exp = TensorflowExperiment(config)
    exp.run()
    exp.save(save_folder)


def resnet(b: int):
    config = TfConfig(
        batch=b,
        epochs=EPOCHS,
        network='resnet'
    )

    exp = TensorflowExperiment(config)
    exp.run()
    exp.save(save_folder)


def run_api() -> Process:
    """Starts the API for setting the metrics"""
    print('Starting api')
    p = Process(target=start_api)
    p.start()
    print('Process started...')
    return p


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', help='Network type for the experiments from [lenet, resnet]')
    args = parser.parse_args()

    net = args.network
    if not net:
        print("Network not set")
        exit(-1)
    elif net not in ('lenet', 'resnet'):
        print('Network', net, 'not among accepted (lenet, resnet)')
        exit(-1)

    api: Process = None
    try:
        # Start the API to collect the metrics
        api = run_api()
        time.sleep(5)

        # based on the arg determine the function
        func = resnet if net == 'resnet' else lenet
        print('Using func', func)

        batches = [128, 64, 32, 16]

        for b in batches:
            func(b)
            time.sleep(10)
    finally:
        print("all experiments finished")
        print(api.pid)
        api.terminate()
        api.join()
