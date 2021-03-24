"""Initial experiments with the lenet network to check the trends of the time with k, batch and parallelism"""

from common.experiment import *
from common.utils import *

output_folder = './tests'


def run_lenet(k: int, batch: int, parallelism: int):
    req = TrainRequest(
        model_type='lenet',
        batch_size=batch,
        epochs=2,
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

    # exp.save(output_folder)


if __name__ == '__main__':
    batches = [64]
    k = [64]
    p = [4]
    # p = [4]

    for b in batches:
        for _k in k:
            for _p in p:
                run_lenet(_k, b, _p)
                time.sleep(20)

    print("all experiments finished")
