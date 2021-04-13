"""Initial experiments with the lenet network to check the trends of the time with k, batch and parallelism"""

from common.experiment import *
from common.metrics import start_api
from multiprocessing import Process
from common.utils import *
import time

import argparse

output_folder = './tests/'

EPOCHS = 30


def run_lenet(k: int, batch: int, parallelism: int):
    req = TrainRequest(
        model_type='lenet',
        batch_size=batch,
        epochs=EPOCHS,
        dataset='mnist',
        lr=0.01,
        function_name='lenet',
        options=TrainOptions(
            default_parallelism=parallelism,
            static_parallelism=True,
            k=k,
            validate_every=1,
            goal_accuracy=100
        )
    )

    exp = KubemlExperiment(get_title(req), req)
    exp.run()
    # exp._fake_history()

    exp.save(output_folder)


def run_resnet(k: int, batch: int, parallelism: int):
    req = TrainRequest(
        model_type='resnet34',
        batch_size=batch,
        epochs=EPOCHS,
        dataset='cifar10',
        lr=0.1,
        function_name='resnet',
        options=TrainOptions(
            default_parallelism=parallelism,
            static_parallelism=True,
            k=k,
            validate_every=1,
            goal_accuracy=100
        )
    )

    exp = KubemlExperiment(get_title(req), req)
    exp.run()
    # print(exp.to_dataframe())
    exp.save(output_folder)


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
        func = run_resnet if net == 'resnet' else run_lenet
        print('Using func', func)

        batches = [16, 32, 64, 128]
        k = [8, 16, 32, -1]
        p = [1, 2, 4, 8]

        for b in batches:
            for _k in k:
                for _p in p:
                    pass
                    func(_k, b, _p)
                    time.sleep(25)
    finally:
        print("all experiments finished")
        print(api.pid)
        api.terminate()
        api.join()
