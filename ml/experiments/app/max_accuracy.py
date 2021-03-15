"""Using a batch of 32, check the max accuracy acheivable in 30 epochs"""
from common.experiment import *
from common.utils import *


def run_lenet(p: int):
    req = TrainRequest(
        model_type='lenet',
        batch_size=32,
        epochs=30,
        dataset='mnist',
        lr=0.01,
        function_name='lenet',
        options=TrainOptions(
            default_parallelism=p,
            static_parallelism=True,
            K=10,
            validate_every=1,
            goal_accuracy=100
        )
    )

    exp = KubemlExperiment(get_title(req), req)
    exp.run()


def run_vgg(p: int):
    req = TrainRequest(
        model_type='vgg',
        batch_size=32,
        epochs=30,
        dataset='cifar100',
        lr=0.01,
        function_name='vgg',
        options=TrainOptions(
            default_parallelism=p,
            static_parallelism=True,
            K=10,
            validate_every=1,
            goal_accuracy=100
        )
    )
    exp = KubemlExperiment(get_title(req), req)
    exp.run()


def run_resnet(p: int):
    req = TrainRequest(
        model_type='resnet',
        batch_size=32,
        epochs=30,
        dataset='cifar10',
        lr=0.01,
        function_name='resnet',
        options=TrainOptions(
            default_parallelism=p,
            static_parallelism=True,
            K=10,
            validate_every=1,
            goal_accuracy=100
        )
    )
    exp = KubemlExperiment(get_title(req), req)
    exp.run()


if __name__ == '__main__':
    parallelism = [2, 4, 8, 16]
    for p in parallelism:
        run_lenet(p)
        time.sleep(5)
        run_resnet(p)
        time.sleep(5)
        run_vgg(p)
