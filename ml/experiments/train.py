"""Initial experiments with the lenet network to check the trends of the time with k, batch and parallelism"""

from common.experiment import *
from common.utils import *
import time

output_folder = './tests/resnet34'


def run_lenet(k: int, batch: int, parallelism: int):
    req = TrainRequest(
        model_type='lenet',
        batch_size=batch,
        epochs=5,
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

    exp.save(output_folder)


def run_resnet(k: int, batch: int, parallelism: int):
    req = TrainRequest(
        model_type='resnet34',
        batch_size=batch,
        epochs=1,
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
    print(exp.to_dataframe())
    exp.save(output_folder)


if __name__ == '__main__':

    # Try to analyze the behavior of the K parameter, the batch size (local)
    # and the parallelism of the functions

    batches = [128, 64, 32]
    k = [8, 16, 64]
    p = [4, 8, 16, 32]

    for b in batches:
        for _k in k:
            for _p in p:
                run_resnet(_k, b, _p)
                # time.sleep(25)

    print("all experiments finished")
