import tensorflow as tf

class Checkpoint:
    def __init__(self, path, max_to_keep = 3, log_step = 10, **kwargs) -> None:
        self.ckpt = tf.train.Checkpoint(step = tf.Variable(1), **kwargs)
        self.manager = tf.train.CheckpointManager(self.ckpt, path, max_to_keep=max_to_keep)
        self.log_step = log_step

    def save(self, force = False):
        self.ckpt.step.assign_add(1)
        if int(self.ckpt.step) % self.log_step == 0 or force:
            self.manager.save()