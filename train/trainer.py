import tensorflow as tf

class Trainer:
    def __init__(self, loss_func, loss, optimizer, metrics, epochs, max_step = -1,
        log_batch = 10, train_tensorboard = None, test_tensorboard = None, checkpoint_manager = None) -> None:
        super().__init__()
        self.loss = loss
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.metrics = metrics
        self.epochs = epochs
        self.log_batch = log_batch
        self.max_step = max_step
        self.train_tensorboard = train_tensorboard
        self.test_tensorboard = test_tensorboard
        self.checkpoint_manager = checkpoint_manager

    @tf.function
    def train_step(self, model, features, labels):
        with tf.GradientTape() as tape:
            predictions = model(features)
            loss = self.loss_func(tf.reshape(labels,[-1]), tf.reshape(predictions,[-1]))
        grads = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        for m in self.metrics:
            m.update_state(tf.reshape(labels, [-1]), tf.reshape(predictions,[-1]))
        self.loss.update_state(loss)

    def get_metric_log(self):
        metric_result = [m.result() for m in self.metrics]
        metric_name = [m.name for m in self.metrics]
        return '\t'.join(['{} = {:.3}'.format(name, result) for name, result in zip(metric_name, metric_result)])

    def get_loss_log(self):
        return 'loss = {:.3}'.format(self.loss.result())

    def reset_state(self):
        self.loss.reset_state()
        for m in self.metrics:
            m.reset_state()

    def log_tensorboard(self, step):
        if not self.train_tensorboard:
            return
        self.train_tensorboard.scalar_log('loss', self.loss.result(), step)
        for m in self.metrics:
            self.train_tensorboard.scalar_log(m.name, m.result(), step)

    def log_info(self, epoch, batch):
        if batch%self.log_batch:
            return
        tf.print("epoch = {}\tbatch = {}\t{}\t{}".format(epoch, batch, self.get_loss_log(), self.get_metric_log()))

    def save_ckpt(self, force = False):
        self.checkpoint_manager.save(force = force)


    def train(self, model, train_data, valid_data = None):
        step = 0
        for epoch in tf.range(1, self.epochs+1):
            batch = 0
            for features, labels in train_data:
                self.train_step(model, features, labels)
                self.log_info(epoch, batch)
                self.log_tensorboard(step)
                self.save_ckpt()
                if self.max_step > 0 and step == self.max_step:
                    self.save_ckpt(force = True)
                    return
                
                
                step += 1
                batch += 1
            self.reset_state()
            