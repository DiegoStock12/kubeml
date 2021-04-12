import argparse
import pandas as pd
import glob
from typing import List

from common.metrics import *
from common.experiment import *

NEURAL_FOLDER = './tests'
HW_FOLDER = './metrics'


def join_df(folder: str) -> pd.DataFrame:
    files = glob.glob(f'{folder}/*.pkl')

    dataframes: List[pd.DataFrame] = []

    for f in files:
        _d = pd.read_pickle(f)
        dataframes.append(_d)

    d = pd.concat(dataframes, ignore_index=True)
    return d


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', help='Type of metrics to be merged from [neural, metrics]')
    parser.add_argument('-o', help='Output name of the pickle file')
    args = parser.parse_args()

    if not args.type or not args.o:
        print("some arguments were not filled")
        exit(-1)

    task = args.type

    if task == 'neural':
        res = join_df(NEURAL_FOLDER)
        print(res)

    elif task == 'metrics':
        res = join_df(HW_FOLDER)
        print(res)


    else:
        print('Error: Type {task} is not recognized, has to be neural or hardware')
        exit(-1)
