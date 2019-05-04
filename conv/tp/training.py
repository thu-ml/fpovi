import os
import tensorflow as tf
import multiprocessing as mp

from tensorpack.tfutils.tower import TowerContext, TowerFuncWrapper
from tensorpack.input_source import QueueInput
from tensorpack.train.tower import TowerTrainer


class BNNTrainer(TowerTrainer):
    
    def __init__(self, input, model):
        super(BNNTrainer, self).__init__()

        cbs = input.setup(model.get_inputs_desc())
        self.register_callback(cbs)

        self.tower_func = TowerFuncWrapper(
            model.build_graph, model.get_inputs_desc())
        
        with TowerContext('', is_training=True):
            self.tower_func(*input.get_input_tensors())
            opt = model.get_optimizer()

        with tf.name_scope('optimize'):
            opt_op = opt.minimize(
                model.inf_loss, var_list=model.inf_vars, name='inf_op')
            if len(model.map_vars) > 0:
                with tf.control_dependencies([opt_op]):
                    opt_op = opt.minimize(
                        model.map_loss, var_list=model.map_vars, name='map_op')
        self.train_op = opt_op


def launch_train_with_config(config):
    """
    Train with a :class:`TrainConfig` and a :class:`Trainer`, to
    present a simple training interface. It basically does the following
    3 things (and you can easily do them by yourself if you need more control):
    1. Setup the input with automatic prefetching heuristics,
       from `config.data` or `config.dataflow`.
    2. Call `trainer.setup_graph` with the input as well as `config.model`.
    3. Call `trainer.train` with rest of the attributes of config.
    Args:
        config (TrainConfig):
        trainer (Trainer): an instance of :class:`BNNTrainer`.
    Example:
    .. code-block:: python
        launch_train_with_config(
            config, SyncMultiGPUTrainerParameterServer(8, ps_device='gpu'))
    """
    assert config.model is not None
    assert config.dataflow is not None

    model = config.model
    input = QueueInput(config.dataflow)
    trainer = BNNTrainer(input, model)

    trainer.train_with_defaults(
        callbacks=config.callbacks,
        monitors=config.monitors,
        session_creator=config.session_creator,
        session_init=config.session_init,
        steps_per_epoch=config.steps_per_epoch,
        starting_epoch=config.starting_epoch,
        max_epoch=config.max_epoch,
        extra_callbacks=config.extra_callbacks)

