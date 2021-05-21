"""trains the tensorflow experiments"""

import argparse
import os
import time
from multiprocessing import Process

from common.tf_experiment import *
from common.metrics import start_api
from tflow.lenet import main as lenet_main
from tflow.resnet34 import main as resnet_main
from tflow.vgg import main as vgg_main

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


def vgg(b: int):
    config = TfConfig(
        batch=b,
        epochs=EPOCHS,
        network='vgg'
    )

    exp = TensorflowExperiment(config, vgg_main)
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
    parser.add_argument('--network', help='Network type for the experiments from [lenet, resnet, vgg]')
    parser.add_argument('-o', help='Folder to save the experiment results to')
    parser.add_argument('-m', help='folder to save the metrics to')
    parser.add_argument('-r', help='Number of replications to run', default=1, type=int)
    args = parser.parse_args()

    net = args.network
    if not net:
        print("Network not set")
        exit(-1)
    elif net not in ('lenet', 'resnet', 'vgg'):
        print('Network', net, 'not among accepted (lenet, resnet, vgg)')
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
        if net == 'resnet':
            func = resnet
            batches = [256, 128, 64, 32]
        elif net == 'lenet':
            func = lenet
            batches = [128, 64, 32, 16]
        elif net == 'vgg':
            func = vgg
            batches = [256, 128, 64]

        print('Using func', func, 'and batches', batches)

        for b in batches:
            print('Using batch', b)
            func(b)
            time.sleep(10)
    finally:
        print("all experiments finished")
        print(api.pid)
        api.terminate()
        api.join()
