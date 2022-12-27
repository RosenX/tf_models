import tensorflow as tf

class TensorBoard:
    def __init__(self, log_dir) -> None:
        self.summary_writer = tf.summary.create_file_writer(log_dir)

    def scalar_log(self, name, data, step):
        with self.summary_writer.as_default():
            tf.summary.scalar(name, data, step)
