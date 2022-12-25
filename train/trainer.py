import tensorflow as tf

class Trainer:
    def __init__(self, loss_func, loss, optimizer, metrics, epochs, log_batch = 10) -> None:
        super().__init__()
        self.loss = loss
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.metrics = metrics
        self.epochs = epochs
        self.log_batch = log_batch

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

    def train(self, data, model):
        for epoch in tf.range(1, self.epochs+1):
            batch = 0
            for features, labels in data:
                self.train_step(model, features, labels)
                if batch%self.log_batch == 0:
                    tf.print("epoch = {} \t batch = {} \t {} \t {}".format(epoch, batch, self.get_loss_log(), self.get_metric_log()))
                batch += 1
            self.reset_state()
            