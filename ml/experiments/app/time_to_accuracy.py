from common.experiment import *
from common.utils import *

"""
Create the experiments for the time to accuracy with multiple networks
(lenet, vgg16 and resnet34). Go for a 90% accuracy, validating every epoch, with
multiple K for testing 

This can also be used to see the time per epoch relationship and the scalability
"""

experiments = {
    'lenet': [],
    'vgg': [],
    'resnet': []
}


def create_request(k: int, p: int, b: int, fun_name: str, dataset: str, goal: float,
                   lr=0.01) -> TrainRequest:
    return TrainRequest(
        model_type=fun_name,
        function_name=fun_name,
        dataset=dataset,
        lr=lr,
        batch_size=b,
        epochs=50,
        options=TrainOptions(
            default_parallelism=p,
            static_parallelism=True,
            K=k,
            validate_every=1,
            goal_accuracy=goal
        )
    )





def run_lenet(k: int, p: int, b: int):
    """Runs the lenet model on the Mnist dataset"""
    req = create_request(k, p, b, 'lenet', 'mnist', 99.0)
    exp = KubemlExperiment(get_title(req), req)

    print(f'Experiment {exp.title} starting...')
    # exp.run()
    print(f'Experiment {exp.title} finished')

    experiments['lenet'].append(exp)


def run_vgg(k: int, p: int, b: int):
    """Runs the vgg model on the Cifar100 dataset"""
    req = create_request(k, p, b, 'vgg', 'cifar100', 80.0)
    exp = KubemlExperiment(get_title(req), req)

    print(f'Experiment {exp.title} starting...')
    # exp.run()
    print(f'Experiment {exp.title} finished')

    experiments['vgg'].append(exp)


def run_resnet(k: int, p: int, b: int):
    """Runs the resnet model on the Cifar10 dataset"""
    req = create_request(k, p, b, 'resnet', 'cifar10', 90.0)
    exp = KubemlExperiment(get_title(req), req)

    print(f'Experiment {exp.title} starting...')
    # exp.run()
    print(f'Experiment {exp.title} finished')

    experiments['resnet'].append(exp)


if __name__ == '__main__':
    # the k parameters to test
    k = [1, 2, 4, 8, 16, 32, 64, -1]

    # the parallelisms to test
    p = [2, 4, 8, 16]

    # the batch sizes (local) to test
    b = [16, 32, 64, 128]

    for _k in k:
        for _p in p:
            for _b in b:
                run_lenet(_k, _p, _b)
