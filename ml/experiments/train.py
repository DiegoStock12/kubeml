"""Initial experiments with the lenet network to check the trends of the time with k, batch and parallelism"""

import argparse
import time
from multiprocessing import Process

from common.experiment import *
from common.metrics import start_api
from common.utils import *

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


def full_parameter_grid(network: str):
    """Generator for the full experiments"""
    if network == 'lenet':
        grid = lenet_grid
    else:
        grid = resnet_grid

    for b in grid['batch']:
        for k in grid['k']:
            for p in grid['parallelism']:
                yield b, k, p


def resume_parameter_grid(network: str, folder: str):
    # find the missing experiments from the folder
    missing = check_missing_experiments(network, folder)
    return missing


def check_folder(path: str) -> bool:
    return os.path.isdir(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', help='Network type for the experiments from [lenet, resnet]')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='''Whether to check for missing experiments and just run those 
                                (best in case of errors preventing the execution of part of the experiments)''')
    parser.add_argument('--folder', help='''if resume is true, path to the folder where 
                                         all the finished experiments reside''')
    parser.add_argument('--dry', dest='dry', action='store_true', help='If true, just print the experiments')
    parser.add_argument('-o', help='Folder to save the experiment results to')
    parser.add_argument('-m', help='folder to save the metrics to')
    parser.add_argument('-r', help='Number of replications to run')

    parser.set_defaults(resume=False, dry=False)
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
        output_folder = args.o

    if args.resume:
        if not args.folder:
            print("Error: Folder not specified with resume")
            exit(-1)

        exps = resume_parameter_grid(net, args.folder)
        output_folder = args.folder
        print("Using", args.folder, 'as output folder')

    else:
        exps = full_parameter_grid(net)

    # if dry, simply print the experiments and return
    if args.dry:
        for e in exps:
            print(e)
        exit(0)

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
        func = run_resnet if net == 'resnet' else run_lenet
        print('Using func', func)

        for batch, k, parallelism in exps:
            print(batch, k, parallelism)
            func(k, batch, parallelism)
            time.sleep(25)

    finally:
        print("all experiments finished")
        print(api.pid)
        api.terminate()
        api.join()
