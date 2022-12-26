import tensorflow as tf
from .. import ModelConfig
from tensorflow.keras import layers
from tf_models.models import BinaryClassifier

class DeepCrossNetworkV2Cofing(ModelConfig):
    def __init__(self, cross_layer_num, input_dim, dense_sizes, mlp_dense_sizes) -> None:
        super().__init__()
        self.cross_layer_num = cross_layer_num
        self.input_dim = input_dim
        self.dense_sizes = dense_sizes
        self.mlp_dense_sizes = mlp_dense_sizes

class CrossBlock(tf.Module):
    def __init__(self, input_dim, cross_layer_num, name):
        super().__init__(name = name)
        self.param = []
        for i in tf.range(cross_layer_num):
            self.param.append((
                tf.Variable(tf.random.normal([input_dim, input_dim])),
                tf.Variable(tf.random.normal([input_dim])),
            ))

    @tf.function
    def __call__(self, x):
        x0 = x
        for w, b in self.param:
            x = tf.multiply(x0, x@w+b)+x
        return x

class DenseBlock(tf.Module):
    def __init__(self, dense_sizes, name):
        super().__init__(name = name)
        self.dense_layers = [layers.Dense(hidden_num, activation = 'relu') for hidden_num in dense_sizes]

    @tf.function
    def __call__(self, x):
        for dense in self.dense_layers:
            x = dense(x)
        return x


class DeepCrossNetworkV2(tf.Module):
    def __init__(self, config, feature, name = 'DeepCrossNetwork'):
        super().__init__(name = name)
        self.cross_block = CrossBlock(config.input_dim, config.cross_layer_num, 'cross_block')
        self.dense_block = DenseBlock(config.dense_sizes, 'dense_block')
        self.mlp_block = DenseBlock(config.mlp_dense_sizes, 'mlp_block')
        self.classifier = BinaryClassifier('classifier')
        self.input_layer = layers.DenseFeatures(feature)

    @tf.function
    def __call__(self, feature_column):
        x = self.input_layer(feature_column)
        dense_output = self.dense_block(x)
        cross_output = self.cross_block(x)
        concat_output = tf.concat([dense_output, cross_output], 1)
        mlp_output = self.mlp_block(concat_output)
        final_output = self.classifier(mlp_output)
        return final_output