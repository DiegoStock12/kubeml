"""trains the tensorflow experiments"""

import argparse
from common.metrics import start_api
from common.experiment import *

from tflow.lenet import main as lenet_main
from tflow.resnet34 import main as resnet_main

from multiprocessing import Process
import time
import os

EPOCHS = 5
save_folder = './tests/tf'


def lenet(b: int):
    config = TfConfig(
        batch=b,
        epochs=EPOCHS,
        network='lenet'
    )

    exp = TensorflowExperiment(config, lenet_main)
    exp.run()
    exp.save(save_folder)


def resnet(b: int):
    config = TfConfig(
        batch=b,
        epochs=EPOCHS,
        network='resnet'
    )

    exp = TensorflowExperiment(config, resnet_main)
    exp.run()
    exp.save(save_folder)


def run_api(path=None) -> Process:
    """Starts the API for setting the metrics"""
    print('Starting api')
    if path is not None:
        p = Process(target=start_api, args=(path,))
    else:
        p = Process(target=start_api)
    p.start()
    print('Process started...')
    return p


def check_folder(path: str) -> bool:
    return os.path.isdir(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', help='Network type for the experiments from [lenet, resnet]')
    parser.add_argument('-o', help='Folder to save the experiment results to')
    parser.add_argument('-m', help='folder to save the metrics to')
    args = parser.parse_args()

    net = args.network
    if not net:
        print("Network not set")
        exit(-1)
    elif net not in ('lenet', 'resnet'):
        print('Network', net, 'not among accepted (lenet, resnet)')
        exit(-1)

    if args.o:
        if not check_folder(args.o):
            print('Given folder does not exist', args.o)
            raise ValueError
        print("Using", args.o, 'as output folder')
        save_folder = args.o

    api: Process = None
    try:

        if args.m:
            if not check_folder(args.m):
                print('Given folder does not exist', args.o)
                raise ValueError
            api = run_api(path=args.m)

        else:
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
