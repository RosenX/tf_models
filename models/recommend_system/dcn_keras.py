import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models, optimizers, losses
from collections import namedtuple

class ModelConfig:
    def __init__(self) -> None:
        pass

class DCNConfig(ModelConfig):
    def __init__(self, cross_layer_num, input_dim, dense_sizes, mlp_dense_sizes) -> None:
        super().__init__()
        self.cross_layer_num = cross_layer_num
        self.input_dim = input_dim
        self.dense_sizes = dense_sizes
        self.mlp_dense_sizes = mlp_dense_sizes

class CrossBlock(layers.Layer):
    def __init__(self, input_dim, cross_layer_num):
        super().__init__()
        self.param = []
        for i in tf.range(cross_layer_num):
            self.param.append((
                tf.Variable(tf.random.normal([input_dim, input_dim])),
                tf.Variable(tf.random.normal([input_dim])),
            ))

    @tf.function
    def call(self, x):
        x0 = x
        for w, b in self.param:
            x = tf.multiply(x0, x@w+b)+x
        return x

class DenseBlock(layers.Layer):
    def __init__(self, dense_sizes):
        super().__init__()
        self.dense_layers = [layers.Dense(hidden_num, activation = 'relu') for hidden_num in dense_sizes]

    @tf.function
    def call(self, x):
        for dense in self.dense_layers:
            x = dense(x)
        return x

class BinaryClassifier(layers.Layer):
    def __init__(self):
        super().__init__()
        self.dense = layers.Dense(1, activation='sigmoid')

    @tf.function
    def call(self, x):
        return self.dense(x)


class DeepCrossNetwork(models.Model):
    def __init__(self, config, feature):
        super(DeepCrossNetwork, self).__init__()
        self.cross_block = CrossBlock(config.input_dim, config.cross_layer_num)
        self.dense_block = DenseBlock(config.dense_sizes)
        self.mlp_block = DenseBlock(config.mlp_dense_sizes)
        self.classifier = BinaryClassifier()
        self.input_layer = layers.DenseFeatures(feature)

    def __call(self, feature_column):
        x = self.input_layer(feature_column)
        dense_output = self.dense_block(x)
        cross_output = self.cross_block(x)
        concat_output = tf.concat([dense_output, cross_output], 1)
        mlp_output = self.mlp_block(concat_output)
        final_output = self.classifier(mlp_output)
        return final_output