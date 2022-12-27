import tensorflow as tf
from tensorflow.keras import metrics

class Trainer:
    def __init__(self, loss_func, optimizer, metrics, 
            tensorboard = None, checkpoint_manager = None, is_valid = False,
            log_step = 10) -> None:
        super().__init__()

        self.state_metrics = []

        self.add_optimizer(optimizer)
        self.add_train_loss(loss_func)
        self.add_train_metric(metrics)

        if is_valid:
            self.add_valid_loss(loss_func)
            self.add_valid_metric(metrics)
        
        self.is_valid = is_valid
        self.log_step = log_step
        
        self.tensorboard = tensorboard
        self.checkpoint_manager = checkpoint_manager
    
    def add_optimizer(self, optimizer):
        self.optimizer = optimizer()

    def add_train_loss(self, loss_func):
        self.loss_func = loss_func()
        self.train_loss = metrics.Mean(name = 'train_{}'.format(loss_func.__name__))
        self.state_metrics.append(self.train_loss)
    
    def add_valid_loss(self, loss_func):
        self.valid_loss_func = loss_func()
        self.valid_loss = metrics.Mean(name = 'valid_{}'.format(loss_func.__name__))
        self.state_metrics.append(self.valid_loss)

    def add_train_metric(self, metrics):
        self.train_metrics = []
        for metric in metrics:
            self.train_metrics.append(metric(name = '{}_{}'.format('train', metric.__name__)))
        self.state_metrics.extend(self.train_metrics)
    
    def add_valid_metric(self, metrics):
        self.valid_metrics = []
        for metric in metrics:
            self.valid_metrics.append(metric(name = '{}_{}'.format('valid', metric.__name__)))
        self.state_metrics.extend(self.valid_metrics)

    @tf.function
    def train_step(self, model, features, labels):
        with tf.GradientTape() as tape:
            predictions = model(features)
            loss = self.loss_func(tf.reshape(labels,[-1]), tf.reshape(predictions,[-1]))
        grads = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        for m in self.train_metrics:
            m.update_state(tf.reshape(labels, [-1]), tf.reshape(predictions,[-1]))
        self.train_loss.update_state(loss)
    
    @tf.function
    def valid_step(self, model, features, labels):
        predictions = model(features)
        loss = self.loss_func(tf.reshape(labels,[-1]), tf.reshape(predictions,[-1]))
        for m in self.valid_metrics:
            m.update_state(tf.reshape(labels, [-1]), tf.reshape(predictions,[-1]))
        self.valid_loss.update_state(loss)

    def get_metric_log(self, metrics):
        metric_result = [m.result() for m in metrics]
        metric_name = [m.name for m in metrics]
        return '\t'.join(['{} = {:.3}'.format(name, result) for name, result in zip(metric_name, metric_result)])

    def get_loss_log(self, loss):
        return '{} = {:.3}'.format(loss.name, loss.result())

    def reset_state(self):
        for m in self.state_metrics:
            m.reset_state()

    def log_tensorboard(self, step):
        if not self.tensorboard:
            return
        
        self.tensorboard.scalar_log('train_loss', self.train_loss.result(), step)

        for m in self.train_metrics:
            self.tensorboard.scalar_log(m.name, m.result(), step)
        
        for m in self.valid_metrics:
            self.tensorboard.scalar_log(m.name, m.result(), step)
        

    def log_info(self, epoch, batch):
        if batch%self.log_batch:
            return

        log_format = "epoch={} \t batch={} \t {} \t {}"
        tf.print(log_format.format(
                epoch, 
                batch, 
                self.get_loss_log(self.train_loss), 
                self.get_metric_log(self.train_metrics)
            ))

    def log_info_by_step(self, step):
        if step%self.log_step:
            return
        if self.is_valid:
            log_format = "Step={:<6} \t {} \t {} \t {} \t {}"
            tf.print(log_format.format(
                    step, 
                    self.get_loss_log(self.train_loss),
                    self.get_metric_log(self.train_metrics),
                    self.get_loss_log(self.valid_loss), 
                    self.get_metric_log(self.valid_metrics)
                ))
        else:
            log_format = "Step: {:<6} \t {} \t {}"
            tf.print(log_format.format(
                    step, 
                    self.get_loss_log(self.train_loss), 
                    self.get_metric_log(self.train_metrics)
                ))

    def save_ckpt(self, force = False):
        self.checkpoint_manager.save(force = force)

    def train(self, model, train_data, epochs):
        step = 0
        for epoch in tf.range(1, epochs+1):
            batch = 0
            for features, labels in train_data:
                self.train_step(model, features, labels)
                self.log_info(epoch, batch)
                self.log_tensorboard(step)
                self.save_ckpt(step == self.max_step)
                step += 1
                batch += 1
            self.reset_state()
    
    def train_valid_by_step(self, model, train_data, valid_data, max_step, shuffle_buffer = 1000):
        train_data = train_data.repeat().shuffle(shuffle_buffer, reshuffle_each_iteration = True)
        valid_data = valid_data.repeat().shuffle(shuffle_buffer, reshuffle_each_iteration = False)
        
        step = 1
        for (train_feature, train_label), (valid_feature, valid_label) in zip(train_data, valid_data):
            self.train_step(model, train_feature, train_label)
            self.valid_step(model, valid_feature, valid_label)

            self.log_info_by_step(step)
            self.log_tensorboard(step)
            self.save_ckpt(step == max_step)

            if step == max_step:
                break

            step += 1