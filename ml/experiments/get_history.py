from common.experiment import *

if __name__ == '__main__':
    req = TrainRequest(
        model_type='example',
        function_name='avg',
        dataset='mnist',
        lr=0.01,
        batch_size=128,
        epochs=1,
        options=TrainOptions(
            default_parallelism=5,
            static_parallelism=True,
            k=16,
            validate_every=0,
            goal_accuracy=90
        )
    )

    exp = KubemlExperiment(title='test experiment', request=req)
    print('running experiment...')
    exp.run()
    # exp.network_id = '5d87491c'

    # exp.wait_for_task_finished()

    print(exp.get_model_history())
    # print(exp.history)
